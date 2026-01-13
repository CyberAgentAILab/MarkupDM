import argparse
from pathlib import Path

import datasets
import huggingface_hub
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

from data_utils import (  # isort: skip
    convert_sample_to_svg,
    decode_class_labels,
    parse_text_completion,
)
from fim_utils import (  # isort: skip
    convert_to_fim,
    revert_from_fim,
)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup
    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model components
    processor, tokenizer, model, vision_model = load_model_components(
        args.model_path, device
    )

    # Load sample from dataset
    sample = load_first_sample(processor, vision_model)
    original_ids = sample["input_ids"]

    # Convert to FIM and generate
    input_ids, target_ids = convert_to_fim(original_ids, tokenizer)
    print(f"Target text: {tokenizer.decode(target_ids)}")

    generated_ids = generate_tokens(model, input_ids, device)

    # Parse generated text completion
    parsed_text, parsed_ids = parse_text_completion(generated_ids, tokenizer)

    if parsed_text != "ParseError":
        print(f"Generated text: {parsed_text}")
    else:
        print("Warning: Failed to parse generated text")

    # Save results
    save_all_results(
        generated_ids=parsed_ids,
        input_ids=input_ids,
        original_ids=original_ids,
        processor=processor,
        vision_model=vision_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    print(f"\nAll results saved to {output_dir}")


def load_model_components(model_path: str, device: str):
    """Load processor, tokenizer, and model."""
    print(f"Loading model components from {model_path}...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Setup font manager
    fonts_path = huggingface_hub.hf_hub_download(
        repo_id="cyberagent/crello",
        revision="5.0.0",
        filename="resources/fonts.pickle",
        repo_type="dataset",
    )
    processor.set_font_manager(fonts_path)

    # Setup tokenizer
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.eval().requires_grad_(False).to(device)
    vision_model = model.vision_model

    return processor, tokenizer, model, vision_model


def load_first_sample(processor, vision_model):
    """Load and preprocess first sample from Crello test split."""
    print("Loading dataset: cyberagent/crello (test split)...")
    dataset = datasets.load_dataset(
        "cyberagent/crello",
        streaming=True,
        split="test",
        revision="5.0.0",
    )
    raw_sample = next(iter(dataset))

    print(f"Preprocessing sample: {raw_sample['id']}...")

    # Decode class labels to strings
    decoded_sample = decode_class_labels(raw_sample, dataset.features)

    # Convert to SVG and extract images
    svg_str, images, filenames = convert_sample_to_svg(decoded_sample)

    # Tokenize: processor will encode images using vision_model
    output = processor(
        svg=svg_str,
        images=images,
        filenames=filenames,
        vision_model=vision_model,
    )

    return output


def generate_tokens(model, input_ids: list[int], device: str) -> np.ndarray:
    """Generate tokens from input."""
    print("Generating...")
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_tensor,
            max_new_tokens=10,
            top_k=0,
            top_p=0.9,
            do_sample=True,
        )

    return generated[0].cpu().numpy()


def save_all_results(
    generated_ids: np.ndarray,
    input_ids: list[int],
    original_ids: torch.Tensor,
    processor,
    vision_model,
    tokenizer,
    output_dir: Path,
) -> None:
    """Save prediction, input, and target results."""
    # Prediction
    pred_ids = revert_from_fim(generated_ids, tokenizer)
    decoded_pred = processor.decode(torch.tensor(pred_ids), vision_model)
    save_result(decoded_pred, output_dir, "pred", processor)

    # Input
    input_only_ids = revert_from_fim(generated_ids[: len(input_ids)], tokenizer)
    decoded_input = processor.decode(torch.tensor(input_only_ids), vision_model)
    save_result(decoded_input, output_dir, "input", processor)

    # Target
    decoded_target = processor.decode(original_ids, vision_model)
    save_result(decoded_target, output_dir, "target", processor)


def save_result(
    decoded: dict,
    output_dir: Path,
    name: str,
    processor,
) -> None:
    """Save decoded result to SVG and PNG."""
    try:
        processor.render(decoded, output_dir / name)
        print(f"  Rendered {name}/")
    except Exception as e:
        print(f"  Warning: Could not render {name}/: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone inference for MarkupDM with FIM and complete design completion"
    )
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument(
        "--model_path",
        type=str,
        default="cyberagent/markupdm",
        help="Path to the pre-trained model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
