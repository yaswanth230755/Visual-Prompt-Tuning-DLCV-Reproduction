#!/usr/bin/env python
"""
Visual Prompt Tuning (VPT) demo: flower recognition with the OFFICIAL VPT code (github.com/KMnP/vpt).

A frozen ViT-B/16 (ImageNet-21k) is combined with the small task-specific parameters that were trained in our
run on Oxford Flowers-102 (prompts + linear head for VPT, bias terms / adapters / head for the baselines).
Only those small files are shipped in checkpoints/; the 413 MB backbone is downloaded once.

Examples
--------
python demo.py --download-weights                       # one-time download of the official ViT-B/16 weights
python demo.py --image my_flower.jpg                    # VPT-deep (best method of this project)
python demo.py --image a.jpg b.jpg --save-figure out.png
python demo.py --folder my_photos --method vpt_shallow
python demo.py --image my_flower.jpg --compare          # all trained methods side by side
python demo.py --list-methods
Works on CPU-only machines (about 1-3 s per image); uses a GPU automatically if one is available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REPO_DIR = ROOT / "third_party" / "vpt_official"      # the authors' repository, unmodified
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "imagenet21k_ViT-B_16.npz"  # file name expected by the official MODEL_ZOO
WEIGHTS_URL = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"
MANIFEST_FILE = ROOT / "checkpoints" / "manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 102


def load_manifest() -> dict:
    """Metadata of the shipped checkpoints (method -> file, learning rate, prompts, measured test accuracy)."""
    if not MANIFEST_FILE.exists():
        sys.exit(f"error: {MANIFEST_FILE} not found - run demo.py from the unpacked project folder.")
    return json.loads(MANIFEST_FILE.read_text())


def download_weights(retries: int = 3) -> None:
    """Fetch the official ViT-B/16 ImageNet-21k checkpoint (about 413 MB) next to the code, with retries."""
    if WEIGHTS_FILE.exists():
        print(f"weights already present: {WEIGHTS_FILE}")
        return
    WEIGHTS_DIR.mkdir(exist_ok=True)
    tmp = WEIGHTS_FILE.with_suffix(".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading {WEIGHTS_URL} (attempt {attempt}/{retries}) ...")
            urllib.request.urlretrieve(WEIGHTS_URL, tmp)
            if tmp.stat().st_size < 300e6:                 # guards against saving an HTML error page
                raise OSError("download too small - not the checkpoint")
            tmp.replace(WEIGHTS_FILE)
            print(f"saved {WEIGHTS_FILE} ({WEIGHTS_FILE.stat().st_size / 1e6:.0f} MB)")
            return
        except Exception as e:                              # noqa: BLE001 - report and retry on any network problem
            last = e
            print(f"  failed: {type(e).__name__}: {e}")
            time.sleep(3)
    sys.exit(f"error: could not download the weights ({last}). Download {WEIGHTS_URL} manually and save it as {WEIGHTS_FILE}.")


def _allow_cpu_only_machines() -> None:
    """The official build_model() asserts NUM_GPUS <= torch.cuda.device_count(), which fails on CPU-only laptops.
    Inference needs no GPU, so on such machines we only make that assertion pass; the official code then runs on CPU."""
    import torch
    if not torch.cuda.is_available():
        torch.cuda.device_count = lambda: 1


class Predictor:
    """One trained VPT / baseline model: official architecture + the shipped task-specific parameters."""

    def __init__(self, method: str):
        manifest = load_manifest()
        if method not in manifest["methods"]:
            sys.exit(f"error: unknown method '{method}'. Choose one of: {', '.join(manifest['methods'])}")
        if not WEIGHTS_FILE.exists():
            sys.exit(f"error: {WEIGHTS_FILE} missing. Run:  python demo.py --download-weights")
        _allow_cpu_only_machines()
        import official_runner as R                          # thin glue around the official classes
        self.R, self.method, self.entry = R, method, manifest["methods"][method]
        repo = R.prepare(str(REPO_DIR))
        cfg = R.make_cfg(repo, method, float(self.entry["base_lr"]), 1e-4, 1, 1, 1, str(ROOT), str(WEIGHTS_DIR),
                         str(ROOT / ".demo_out"), num_tokens=int(self.entry["prompts"]), workers=0, num_classes=NUM_CLASSES)
        self.model, self.device = R.load_trained(cfg, str(ROOT / "checkpoints" / self.entry["file"]))
        self.transform = R.official_test_transform(224)      # the official evaluation transform (resize 256, crop 224)
        from flower_names import FLOWER_NAMES
        self.names = FLOWER_NAMES

    def predict(self, paths: list[Path], topk: int = 5) -> list[dict]:
        import torch
        from PIL import Image
        batch = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in paths])
        logits, _ = self.R.features_and_logits(self.model, self.device, batch)
        probs = torch.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(min(topk, NUM_CLASSES), dim=-1)
        return [{"path": str(p), "top": [(self.names[int(i)], float(q), int(i)) for q, i in zip(tp, ti)]}
                for p, tp, ti in zip(paths, top_p, top_i)]

    def describe(self) -> str:
        e = self.entry
        return (f"{e['label']} | Oxford Flowers-102 test accuracy {e['test_acc']:.2f}% | "
                f"tuned parameters {e['tuned_percent']:.3f}% of the backbone")


def is_readable_image(path: Path) -> bool:
    """True when Pillow can open the file as an image (guards against corrupt files, HTML pages saved as .jpg, ...)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:                                       # noqa: BLE001 - any decoding problem means "not usable"
        return False


def list_images(folder: Path) -> list[Path]:
    """Readable images of a folder, sorted by name; unreadable files are skipped with a visible warning."""
    found = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    good = [p for p in found if is_readable_image(p)]
    for p in found:
        if p not in good:
            print(f"  warning: skipping '{p.name}' - not a readable image file")
    return good


def find_user_images(extra_dirs: list[str] | None = None) -> list[Path]:
    """Images supplied by the examiner, searched in this order: $VPT_DEMO_IMAGES, ./sir_images, extra_dirs, then any
    attached Kaggle dataset (/kaggle/input) that is not this project. Returns [] when no readable image is found."""
    import os
    candidates = [os.environ.get("VPT_DEMO_IMAGES"), str(ROOT / "sir_images"), *(extra_dirs or [])]
    for c in candidates:
        if c and Path(c).is_dir():
            imgs = list_images(Path(c))
            if imgs:
                return imgs
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        folders = sorted({p.parent for p in kaggle_input.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and "third_party" not in p.parts})
        for f in folders:
            imgs = list_images(f)
            if imgs:
                return imgs
    return []


def collect_images(images: list[str], folder: str | None) -> list[Path]:
    paths = [Path(p) for p in images or []]
    if folder:
        d = Path(folder)
        if not d.is_dir():
            sys.exit(f"error: folder not found: {d}")
        paths += list_images(d)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit("error: image not found: " + ", ".join(map(str, missing)))
    unreadable = [p for p in paths if not is_readable_image(p)]
    if unreadable:
        sys.exit("error: not a readable image file: " + ", ".join(map(str, unreadable)))
    if not paths:
        sys.exit("error: give --image FILE [FILE ...] or --folder DIR (see --help)")
    return paths


LOW_CONFIDENCE = 0.50      # below this the image is probably not one of the 102 flower categories


def low_confidence_note(top1_prob: float) -> str:
    """The model can only answer with one of the 102 trained flower classes, even for a non-flower image."""
    return (f"  note: confidence {100 * top1_prob:.0f}% is low - the image may not show one of the 102 Oxford flower categories."
            if top1_prob < LOW_CONFIDENCE else "")


def plot_results(results: list[dict], title: str = ""):
    """One row per image: the picture on the left, the top-k probabilities as bars on the right."""
    import matplotlib.pyplot as plt
    from PIL import Image
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.6 * n), squeeze=False, gridspec_kw={"width_ratios": [1, 1.6]})
    for row, r in zip(axes, results):
        row[0].imshow(Image.open(r["path"]).convert("RGB")); row[0].axis("off")
        row[0].set_title(Path(r["path"]).name[:28], fontsize=8)
        names = [t[0] for t in r["top"]][::-1]
        probs = [100 * t[1] for t in r["top"]][::-1]
        colors = ["#d62728" if i == len(names) - 1 else "#8da0cb" for i in range(len(names))]
        row[1].barh(names, probs, color=colors); row[1].set_xlim(0, 100); row[1].set_xlabel("probability (%)", fontsize=8)
        row[1].tick_params(labelsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def print_predictions(pred: Predictor, results: list[dict]) -> None:
    print(f"\n[{pred.method}] {pred.describe()}")
    for r in results:
        print(f"\n  image: {r['path']}")
        for rank, (name, prob, idx) in enumerate(r["top"], 1):
            print(f"    {rank}. {name:<28s} {100 * prob:5.1f}%   (class {idx})")
        note = low_confidence_note(r["top"][0][1])
        if note:
            print(note)


def save_figure(results_by_method: dict[str, list[dict]], out: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not installed - skipping --save-figure (pip install matplotlib)")
        return
    for method, results in results_by_method.items():
        path = out if len(results_by_method) == 1 else str(Path(out).with_name(f"{Path(out).stem}_{method}{Path(out).suffix}"))
        plot_results(results, title=method).savefig(path, dpi=150, bbox_inches="tight")
        print(f"figure saved: {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VPT flower demo (official VPT code, Oxford Flowers-102)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--image", nargs="+", help="one or more image files")
    ap.add_argument("--folder", help="classify every image in this folder")
    ap.add_argument("--method", default="vpt_deep", help="vpt_deep (default), vpt_shallow, linear, bias, adapter")
    ap.add_argument("--compare", action="store_true", help="run every shipped method on the same images")
    ap.add_argument("--topk", type=int, default=5, help="number of predictions to print (default 5)")
    ap.add_argument("--save-figure", metavar="PNG", help="also save a figure with the top-1 predictions")
    ap.add_argument("--download-weights", action="store_true", help="download the official ViT-B/16 ImageNet-21k weights and exit")
    ap.add_argument("--list-methods", action="store_true", help="show the shipped checkpoints and exit")
    a = ap.parse_args(argv)

    if a.download_weights:
        download_weights()
        return 0
    manifest = load_manifest()
    if a.list_methods:
        for m, e in manifest["methods"].items():
            print(f"{m:12s} {e['label']:24s} test acc {e['test_acc']:.2f}%   tuned {e['tuned_percent']:.3f}%")
        return 0
    paths = collect_images(a.image, a.folder)
    methods = list(manifest["methods"]) if a.compare else [a.method]
    results_by_method = {}
    for m in methods:
        pred = Predictor(m)
        results_by_method[m] = pred.predict(paths, a.topk)
        print_predictions(pred, results_by_method[m])
        del pred                                            # free the backbone before loading the next model
    if a.save_figure:
        save_figure(results_by_method, a.save_figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
