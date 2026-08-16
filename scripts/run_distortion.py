import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.distortion.d0 import D0Gaussian
import yaml

CFG = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
DEFAULT_FWHM_PX = CFG["degradation"]["diffraction_fwhm_px"]


# Реестр: имя из атрибута класса -> сам класс.
# Добавить D1 = дописать его в этот список, argparse подхватит автоматически.
GENERATORS = {cls.name: cls for cls in [D0Gaussian]}


def parse_args():
    p = argparse.ArgumentParser(description="Прогон одного генератора деградации")
    p.add_argument("generator", choices=sorted(GENERATORS),
                   help="какой генератор запустить")
    p.add_argument("--input", type=Path, required=True,
                   help="путь к исходному изображению")
    p.add_argument("--d-over-r0", type=float, required=True,
                   help="отношение D/r0")
    p.add_argument("--fwhm", type=float, default=DEFAULT_FWHM_PX,
                   help="дифракционная FWHM в пикселях")
    p.add_argument("--seed", type=int, default=0,
                   help="seed для rng (D0 игнорирует, нужен для D1+)")
    p.add_argument("--output", type=Path, default=None,
                   help="путь для сохранения; по умолчанию имя собирается из параметров")
    return p.parse_args()


def load_image(path):
    """uint8 [0,255] -> float32 [0,1], форма (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def save_image(arr, path):
    """float32 [0,1] -> uint8 [0,255]. round() обязателен, иначе картинка темнеет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(out).save(path)


def build_output_path(args):
    stem = args.input.stem
    return Path("scripts") / (
        f"{stem}_{args.generator}_dr{args.d_over_r0:g}_seed{args.seed}.png"
    )

def main():
    args = parse_args()

    img = load_image(args.input)

    degradation = GENERATORS[args.generator](diffraction_fwhm_px=args.fwhm)
    rng = np.random.default_rng(args.seed)
    out = degradation(img, args.d_over_r0, rng)

    out_path = args.output or build_output_path(args)
    save_image(out, out_path)

    print(f"{args.generator}: {img.shape} -> {out.shape}, "
          f"D/r0={args.d_over_r0}, сохранено в {out_path}")


if __name__ == "__main__":
    main()
