import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def main(image_path: str, output_path: str | None, model_path: str):
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = AutoImageProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)

    # Output path
    if output_path is None:
        _path = Path(image_path)
        _output_path = _path.with_name(_path.stem + "_recon.png")
    else:
        _output_path = Path(output_path)
    _output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get image size
    img = Image.open(image_path).convert("RGBA")

    # Image reconstruction
    example = image_processor(img)
    image = example["image"].unsqueeze(0)
    image = image.to(dtype=model.dtype, device=model.device)
    with torch.inference_mode():
        recon, _ = model.model(image)
    recon_img = image_processor.postprocess(recon[0].float().cpu())
    recon_img.resize(img.size).save(_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the reconstructed image. Default: {image_path}_recon.png",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cyberagent/ldm-vq-f16-rgba",
        help="Path to the pre-trained model.",
    )
    args = parser.parse_args()
    main(args.image_path, args.output_path, args.model_path)
