from argparse import ArgumentParser
from pathlib import Path
import io

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


def corrupt(image: Image.Image, name: str, seed: int, severity: float) -> Image.Image:
    image = image.convert("RGB")
    if name == "noise":
        array = np.asarray(image, dtype=np.float32)
        rng = np.random.default_rng(seed)
        return Image.fromarray(np.clip(array + rng.normal(0, severity, array.shape), 0, 255).astype(np.uint8))
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if name == "lowlight":
        return ImageEnhance.Brightness(image).enhance(severity)
    raise ValueError(name)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["noise:25", "blur:1.5", "jpeg:20", "lowlight:0.45"])
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    output_rows = []
    for spec in args.variants:
        name, severity_text = spec.split(":", 1)
        severity = float(severity_text)
        variant = f"{name}_{severity_text.replace('.', 'p')}"
        variant_root = args.output_root / variant
        for row_index, row in frame.iterrows():
            source = Path(row["path"])
            destination = variant_root / source.parent.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                corrupt(image, name, seed=1000 + row_index, severity=severity).save(destination)
            output_rows.append(
                {
                    "path": str(destination),
                    "label": int(row["label"]),
                    "domain": f"fer2013_{variant}",
                    "split": "test",
                }
            )
    output_manifest = args.output_root / "manifests" / "fer2013_all_corruptions.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output_manifest, index=False)
    print(output_manifest)


if __name__ == "__main__":
    main()
