"""Smoke test of the demo on random images. Needs weights/imagenet21k_ViT-B_16.npz (real or any file in that format).

Run:  python tests/test_demo_smoke.py
Checks, for every shipped checkpoint: the saved prompts/head load into the official model without unexpected keys,
predictions have valid class names, probabilities are in [0, 1] and sorted, and the CLI runs end to end.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import demo  # noqa: E402
from flower_names import FLOWER_NAMES  # noqa: E402


def make_images(folder: Path, n: int = 3) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = folder / f"random_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (300 + 40 * i, 400, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def main() -> int:
    if not demo.WEIGHTS_FILE.exists():
        print(f"SKIP: {demo.WEIGHTS_FILE} not found (run: python demo.py --download-weights)")
        return 0
    manifest = demo.load_manifest()
    with tempfile.TemporaryDirectory() as tmp:
        imgs = make_images(Path(tmp))
        for method in manifest["methods"]:
            pred = demo.Predictor(method)
            out = pred.predict(imgs, topk=5)
            assert len(out) == len(imgs)
            for r in out:
                probs = [t[1] for t in r["top"]]
                assert len(r["top"]) == 5 and all(0.0 <= p <= 1.0 for p in probs) and probs == sorted(probs, reverse=True)
                assert all(t[0] in FLOWER_NAMES for t in r["top"])
            print(f"[PASS] {method}: checkpoint loads, 3 images -> valid top-5 predictions")
            del pred
        rc = demo.main(["--image", str(imgs[0]), "--method", "vpt_deep", "--topk", "3", "--save-figure", str(Path(tmp) / "f.png")])
        assert rc == 0 and (Path(tmp) / "f.png").exists()
        print("[PASS] CLI end to end (+ figure file)")
    print("all smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
