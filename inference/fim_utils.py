"""Utility functions for simple inference script."""

import re
from typing import Literal

import numpy as np
from transformers import PreTrainedTokenizerBase


def get_fim_token(
    tokenizer: PreTrainedTokenizerBase,
    fim_type: Literal["prefix", "middle", "suffix"],
) -> int:
    """Get FIM token ID from tokenizer."""
    token_id = tokenizer.convert_tokens_to_ids(f"<fim_{fim_type}>")
    if token_id is None:
        token_id = tokenizer.convert_tokens_to_ids(f"<|fim_{fim_type}|>")
    assert token_id is not None, f"{fim_type} token not found"
    return token_id


def convert_to_fim(
    original_ids: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    target_text_index: int = 3,
) -> tuple[list[int], list[int]]:
    """
    Convert token IDs to FIM format by extracting text content.

    Args:
        original_ids: Token IDs (numpy array or torch tensor)
        tokenizer: Tokenizer
        target_text_index: Index of the text element to use as target (0-indexed, default=3)

    Returns:
        (input_ids, target_ids) tuple where input_ids is in PSM format
    """
    # Get FIM token IDs
    eos_id = tokenizer.eos_token_id
    pre_id = get_fim_token(tokenizer, "prefix")
    mid_id = get_fim_token(tokenizer, "middle")
    suf_id = get_fim_token(tokenizer, "suffix")
    img_tok_id = tokenizer.convert_tokens_to_ids("<image_token>")

    # Convert to numpy array if it's a torch tensor
    if hasattr(original_ids, "cpu"):
        original_ids = original_ids.cpu().numpy()
    original_ids = np.array(original_ids)

    # Check if starts/ends with EOS
    starts_with_eos = original_ids[0] == eos_id
    if starts_with_eos:
        original_ids = original_ids[1:]

    ends_with_eos = original_ids[-1] == eos_id
    if ends_with_eos:
        original_ids = original_ids[:-1]

    # Make image tokens decodable
    original_ids_np = np.array(original_ids)
    image_mask = original_ids_np >= len(tokenizer)
    dense_image_ids = original_ids_np[image_mask] - len(tokenizer)
    original_ids_np[image_mask] = img_tok_id

    # Decode document
    document_str = tokenizer.decode(original_ids_np, skip_special_tokens=False)

    # Find text elements
    pattern = r"<text[^>]*[^/]>(.*?)</text>"
    matches = list(re.finditer(pattern, document_str, re.MULTILINE | re.DOTALL))
    assert len(matches) > 0, f"No text element found in SVG. Pattern: {pattern}"

    # Use the specified text element index if available, otherwise use the last one
    target_index = min(target_text_index, len(matches) - 1)
    match = matches[target_index]

    # Extract prefix, middle, suffix
    i, j = match.span(1)  # group 1 is the text content
    prefix_str = document_str[:i]
    middle_str = document_str[i:j]
    suffix_str = document_str[j:]

    # Tokenize strings
    prefix = np.array(tokenizer.encode(prefix_str, return_tensors="np")[0])
    middle = np.array(tokenizer.encode(middle_str, return_tensors="np")[0])
    suffix = np.array(tokenizer.encode(suffix_str, return_tensors="np")[0])

    # Reorder dense image IDs
    num_pre_img = np.sum(prefix == img_tok_id)
    num_mid_img = np.sum(middle == img_tok_id)
    _image_ids = np.concatenate(
        [
            dense_image_ids[:num_pre_img],  # Prefix
            dense_image_ids[num_pre_img + num_mid_img :],  # Suffix
            dense_image_ids[num_pre_img : num_pre_img + num_mid_img],  # Middle
        ]
    )

    # PSM (Prefix-Suffix-Middle)
    document = [pre_id, *prefix, suf_id, *suffix, mid_id, *middle]

    # Merge image tokens
    input_ids = np.array(document)
    image_mask = input_ids == img_tok_id
    input_ids[image_mask] = _image_ids + len(tokenizer)

    # Split into input and target
    i_mid = np.where(input_ids == mid_id)[0][0]
    input_ids_fim = input_ids[: i_mid + 1]
    target_ids = input_ids[i_mid + 1 :]

    if starts_with_eos:
        input_ids_fim = np.concatenate([[eos_id], input_ids_fim])

    return input_ids_fim.tolist(), target_ids.tolist()


def revert_from_fim(
    input_ids: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
) -> np.ndarray:
    """Revert from FIM format (PSM: Prefix-Suffix-Middle) to normal order."""
    input_ids = input_ids.copy()

    # Get FIM token IDs
    eos_id = tokenizer.eos_token_id
    pre_id = get_fim_token(tokenizer, "prefix")
    mid_id = get_fim_token(tokenizer, "middle")
    suf_id = get_fim_token(tokenizer, "suffix")

    # Check if input is in FIM format
    assert pre_id in input_ids, "FIM_PREFIX token is not found"
    assert mid_id in input_ids, "FIM_MIDDLE token is not found"
    assert suf_id in input_ids, "FIM_SUFFIX token is not found"

    i_pre = np.where(input_ids == pre_id)[0]
    i_mid = np.where(input_ids == mid_id)[0]
    i_suf = np.where(input_ids == suf_id)[0]
    msg = f"Invalid FIM format: {i_mid.size=}, {i_suf.size=}, {i_pre.size=}"
    assert i_mid.size == i_suf.size == i_pre.size, msg

    while i_pre.size > 0:
        # Process FIM tokens from right to left
        i_pre, i_mid, i_suf = i_pre[-1], i_mid[-1], i_suf[-1]
        msg = f"Invalid FIM format: {i_pre=}, {i_suf=}, {i_mid=}"
        assert i_pre < i_suf < i_mid, msg

        # Find EOS position
        i_eos = np.where(input_ids[i_mid:] == eos_id)[0] + i_mid
        i_eos = len(input_ids) if i_eos.size == 0 else i_eos[0]

        # Reorder: [before_prefix, prefix, middle, suffix, after_eos]
        input_ids = np.array(
            [
                *input_ids[:i_pre],  # before prefix
                *input_ids[i_pre + 1 : i_suf],  # prefix
                *input_ids[i_mid + 1 : i_eos],  # middle
                *input_ids[i_suf + 1 : i_mid],  # suffix
                *input_ids[i_eos:],  # after suffix
            ]
        )

        # Update indices for next iteration
        i_pre = np.where(input_ids == pre_id)[0]
        i_mid = np.where(input_ids == mid_id)[0]
        i_suf = np.where(input_ids == suf_id)[0]

    return input_ids
