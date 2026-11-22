#!/usr/bin/env python
"""Consistency check of this package (no GPU / torch needed):  python tools/verify_package.py

1. every required file is present          2. the official code in third_party/ is byte-identical to the original
3. README numbers == results/final_results.csv == checkpoints/manifest.json == per-run result files
4. the 5 figures and the executed training notebook are present, and the notebook has no error outputs
5. all team members are listed in the README          6. demo.py / official_runner.py compile
"""
import csv
import hashlib
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = []


def chk(name, ok, info=""):
    res.append((name, bool(ok), info))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


req = ["README.md", "NOTICE.md", "requirements.txt", "demo.py", "demo.ipynb", "official_runner.py", "flower_names.py",
       "checkpoints/manifest.json", "results/final_results.csv", "results/summary.md", "results/paper_conformance.json",
       "results/official_code_integrity.json", "third_party/vpt_official/LICENSE", "MANIFEST.sha256",
       "notebooks/01_train_evaluate_kaggle_executed.ipynb", "tests/test_demo_smoke.py", "tests/test_windows_path_shim.py"]
for f in req:
    chk(f"file present: {f}", (ROOT / f).exists())

# 2) third-party code unchanged
for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
    digest, rel = line.split("  ", 1)
    p = ROOT / rel
    if not (p.exists() and sha(p) == digest):
        chk(f"official file unchanged: {rel}", False)
chk("official repo: every file in MANIFEST.sha256 is byte-identical", not any(n.startswith("official file unchanged") and not ok for n, ok, _ in res))
chk("official repo: no extra files", len([p for p in (ROOT / "third_party/vpt_official").rglob("*") if p.is_file() and "__pycache__" not in p.parts])
    == len((ROOT / "MANIFEST.sha256").read_text().splitlines()))

# 3) numbers
manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
rows = {r["method_label"]: r for r in csv.DictReader(open(ROOT / "results/final_results.csv"))}
readme = (ROOT / "README.md").read_text()
for m, e in manifest["methods"].items():
    csvrow = next(r for r in rows.values() if r["method_label"].lower().startswith(e["label"].lower().split(" ")[0]))
    chk(f"{m}: manifest test acc == final_results.csv", abs(e["test_acc"] - float(csvrow["test_%"])) < 5e-3, f"{e['test_acc']:.2f}")
    chk(f"{m}: README shows {e['test_acc']:.2f}", f"{e['test_acc']:.2f}" in readme)
    run = json.loads((ROOT / "results/runs_json" / f"{e['source_run']}.json").read_text())
    chk(f"{m}: run file {e['source_run']} test acc == manifest", abs(100 * run["test_top1"] - e["test_acc"]) < 5e-3)
    chk(f"{m}: checkpoint file exists", (ROOT / "checkpoints" / e["file"]).exists())
chk("README shows partial-1 and full-fine-tuning rows", "Partial-1" in readme and "Full fine-tuning" in readme)

# 4) figures / notebook
for f in ("accuracy_overview", "training_curves", "ablation_prompt_length", "tsne", "demo_predictions"):
    chk(f"figure {f}.png", (ROOT / "results/figures" / f"{f}.png").exists())
nb = json.loads((ROOT / "notebooks/01_train_evaluate_kaggle_executed.ipynb").read_text())
errs = sum(1 for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
chk("executed training notebook: 0 error outputs", errs == 0, f"{errs} errors")
chk("conformance 16/16", all(x["passed"] for x in json.loads((ROOT / "results/paper_conformance.json").read_text())))
chk("official code integrity: no modified file", json.loads((ROOT / "results/official_code_integrity.json").read_text())["files_modified"] == [])

# 5) team + course
for roll in ("B230857CS", "B230673CS", "B230394CS", "B230755CS", "B230152CS"):
    chk(f"README lists {roll}", roll in readme)
chk("README names the course and instructor", "CS4046E" in readme and "Santosh Kumar Behera" in readme)
chk("README names the paper and official repo", "Visual Prompt Tuning" in readme and "github.com/KMnP/vpt" in readme)

# 5b) demo notebook / Windows shim / doc facts
dn = json.loads((ROOT / "demo.ipynb").read_text())
chk("demo.ipynb is self-contained (embedded project) and has no stored outputs",
    any("PROJECT_ZIP_B64" in "".join(c["source"]) for c in dn["cells"]) and not any(c.get("outputs") for c in dn["cells"] if c["cell_type"] == "code"))
chk("official_runner.py contains the Windows path shim", "posixpath.join" in (ROOT / "official_runner.py").read_text())
chk("README states the real weights size (413 MB)", "413 MB" in readme and "393" not in readme)

# 6) compile
for f in ("demo.py", "official_runner.py", "flower_names.py", "tests/test_demo_smoke.py", "tests/test_windows_path_shim.py"):
    try:
        py_compile.compile(str(ROOT / f), doraise=True); ok = True
    except py_compile.PyCompileError:
        ok = False
    chk(f"compiles: {f}", ok)

w = max(len(n) for n, _, _ in res)
bad = [n for n, ok, _ in res if not ok]
for n, ok, i in res:
    print(f"[{'PASS' if ok else 'FAIL'}] {n.ljust(w)} {i}")
print(f"\n{len(res) - len(bad)}/{len(res)} checks passed" + (f"   FAILED: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
