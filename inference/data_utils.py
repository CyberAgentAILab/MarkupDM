"""Utility functions for processing Crello dataset samples."""

import html
import re
from typing import Any, Callable, Literal

import datasets
import numpy as np
import svg
from PIL import Image

from fim_utils import get_fim_token  # isort: skip


def decode_class_labels(example: dict, features: datasets.Features) -> dict:
    """
    Decode class labels to strings (following vistacreate.py logic).

    Args:
        example: Raw dataset sample with numeric class labels
        features: Dataset features containing ClassLabel definitions

    Returns:
        Sample with class labels converted to strings
    """

    def _get_decode_fn(feature: Any) -> Callable:
        if isinstance(feature, datasets.ClassLabel):
            return feature.int2str  # type: ignore
        elif isinstance(feature, datasets.Sequence):
            return _get_decode_fn(feature.feature)
        return lambda x: x

    output = {}
    for key, feature in features.items():
        decode_fn = _get_decode_fn(feature)
        output[key] = decode_fn(example[key])
    return output


def get_text(text: str, capitalize: bool) -> str:
    """Get escaped text with capitalization."""
    text = text.upper() if capitalize else text
    return html.escape(text)


def get_text_anchor_and_x(
    text_align: str,
    left: float,
    width: float,
) -> tuple[Literal["start", "middle", "end"], int]:
    """Get text anchor and x position based on alignment."""
    text_align = "left" if text_align == "justify" else text_align

    text_anchor: Literal["start", "middle", "end"]
    if text_align == "left":
        text_anchor = "start"
        x = left
    elif text_align == "center":
        text_anchor = "middle"
        x = left + width / 2.0
    elif text_align == "right":
        text_anchor = "end"
        x = left + width
    else:
        raise ValueError(f"Unknown text_align: {text_align}")

    return text_anchor, round(x)


def get_transform(
    angle: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> list[svg.Transform] | None:
    """Get transform for rotation."""
    if angle == 0.0:
        return None

    x_center = round(left + width / 2.0)
    y_center = round(top + height / 2.0)
    angle_rounded = round(angle)

    return [svg.Rotate(angle_rounded, x_center, y_center)]


def convert_sample_to_svg(sample: dict) -> tuple[str, list[Image.Image], list[str]]:
    """
    Convert raw Crello sample to SVG string with images.

    Includes both text and image elements, matching MarkupDM processor expectations.
    This implementation follows the logic from vistacreate.py for high fidelity.

    Args:
        sample: Raw Crello dataset sample (should have class labels already decoded to strings)

    Returns:
        Tuple of (svg_string, images, filenames)
    """
    svg_elements: list[svg.Element] = []
    images: list = []
    filenames: list[str] = []

    for i in range(sample["length"]):
        elem_type = sample["type"][i]
        left = sample["left"][i]
        top = sample["top"][i]
        width = sample["width"][i]
        height = sample["height"][i]
        angle = sample["angle"][i]
        opacity = sample["opacity"][i]

        # Format opacity
        opacity_value = round(opacity, 2) if opacity != 1.0 else None

        # Get transform
        transform = get_transform(angle, left, top, width, height)

        if elem_type == "TextElement":
            # Font weight and style
            _bold = sample["font_bold"][i]
            _bold = _bold if isinstance(_bold, bool) else any(_bold)
            font_weight: Literal["bold"] | None = "bold" if _bold else None

            _italic = sample["font_italic"][i]
            _italic = _italic if isinstance(_italic, bool) else any(_italic)
            font_style: Literal["italic"] | None = "italic" if _italic else None

            # Position
            y = top
            text_align = sample["text_align"][i]
            text_anchor, x = get_text_anchor_and_x(text_align, left, width)

            # Color
            text_color = sample["text_color"][i]
            if isinstance(text_color, list):
                fill = text_color[0]
            else:
                fill = text_color
            fill = fill if fill != "rgba(0, 0, 0, 1)" else None

            # Letter spacing
            letter_spacing_val = round(sample["letter_spacing"][i], 2)
            letter_spacing = letter_spacing_val if letter_spacing_val != 0.0 else None

            # Process text with capitalization (following vistacreate.py to_elements logic)
            processed_text = get_text(sample["text"][i], sample["capitalize"][i])

            # Split text by line
            text_line = sample["text_line"][i]
            if len(text_line) > 0:
                # For multi-line text, split the processed text by line indices
                texts = [""] * (max(text_line) + 1)
                for i_line, char in zip(text_line, processed_text):
                    texts[i_line] += char
            else:
                texts = [processed_text]

            # Generate text elements for each line
            line_height = sample["line_height"][i]
            font_size = sample["font_size"][i]
            for text in texts:
                svg_element = svg.Text(
                    text=text,
                    font_family=sample["font"][i],
                    font_size=round(font_size),
                    font_weight=font_weight,
                    font_style=font_style,
                    text_anchor=text_anchor if text_anchor != "start" else None,
                    x=round(x),
                    y=round(y),
                    fill=fill,
                    letter_spacing=letter_spacing,
                    transform=transform,
                    opacity=opacity_value,
                )
                svg_elements.append(svg_element)
                y += font_size * line_height

        else:
            # Image element
            image_id = len(images)
            filename = f"{image_id:03d}.png"
            images.append(sample["image"][i])
            filenames.append(filename)
            svg_element = svg.Image(
                href=filename,
                x=round(left),
                y=round(top),
                width=round(width),
                height=round(height),
                transform=transform,
                opacity=opacity_value,
            )
            svg_elements.append(svg_element)

    # SVG
    canvas_width = sample["canvas_width"]
    canvas_height = sample["canvas_height"]
    view_box = svg.ViewBoxSpec(0, 0, canvas_width, canvas_height)
    svg_text = svg.SVG(
        width=canvas_width,
        height=canvas_height,
        viewBox=view_box,
        elements=svg_elements,
    ).as_str()

    return svg_text, images, filenames


def extract_images_from_sample(sample: dict) -> tuple[list, list]:
    """
    Extract PIL images from Crello sample.

    Args:
        sample: Raw Crello dataset sample

    Returns:
        (images, filenames) tuple where images are PIL.Image objects
    """
    images = []
    filenames = []

    for i in range(sample["length"]):
        elem_type = sample["type"][i]
        if elem_type != 2:  # Not TextElement
            image = sample["image"][i]
            images.append(image)
            filenames.append(f"{len(images) - 1:03d}.png")

    return images, filenames


def parse_text_completion(
    generated_ids: np.ndarray,
    tokenizer,
) -> tuple[str, np.ndarray]:
    """
    Parse text completion from FIM format.

    Extracts the generated middle section between <fim_middle> and </text> closing tag.

    Args:
        generated_ids: Full token sequence including input (FIM format)
        tokenizer: Tokenizer instance

    Returns:
        (parsed_str, parsed_ids): Parsed text string and token IDs
    """

    # Find <fim_middle> token position
    mid_id = get_fim_token(tokenizer, "middle")
    mid_positions = np.where(generated_ids == mid_id)[0]

    if len(mid_positions) == 0:
        # No FIM middle token found
        return "ParseError", generated_ids

    i_mid = mid_positions[0]
    input_ids = generated_ids[
        : i_mid + 1
    ]  # Everything up to and including <fim_middle>
    pred_ids = generated_ids[i_mid + 1 :]  # Generated part after <fim_middle>

    # Decode generated part
    pred_str = tokenizer.decode(pred_ids)

    # Parse until </text> closing tag
    pattern = r"[^<]+"  # Match any characters except '<'
    matched = re.match(pattern, pred_str, re.DOTALL)

    if matched is None:
        # Parse failed
        parsed_str = "ParseError"
        parsed_ids = input_ids
    else:
        # Successfully parsed
        parsed_str = matched.group()
        # Re-encode to get clean token IDs
        parsed_tokens = tokenizer.encode(parsed_str, return_tensors="np")[0]
        parsed_ids = np.concatenate([input_ids, parsed_tokens])

    return parsed_str, parsed_ids
