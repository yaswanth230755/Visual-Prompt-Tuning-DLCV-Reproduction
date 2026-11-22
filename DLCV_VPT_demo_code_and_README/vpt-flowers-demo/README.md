# Visual Prompt Tuning (VPT) — reproduction with the official code and flower-recognition demo

**CS4046E Deep Learning for Computer Vision** · Instructor: **Dr. Santosh Kumar Behera**, Assistant Professor, CSE, NIT Calicut (skbehera@nitc.ac.in)

| Team member | Roll number |
|---|---|
| Atluri Venkata Sai Vignesh Chowdary | B230857CS |
| Malepati Venkata Gowrish | B230673CS |
| Kuruva Manorath | B230394CS |
| Adimulam Yaswanth Veera Nagesh | B230755CS |
| Akkala Pavan Reddy | B230152CS |

**Paper followed:** M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, S.-N. Lim, *Visual Prompt Tuning*, ECCV 2022 (arXiv:2203.12119).
**Official code (used unmodified):** https://github.com/KMnP/vpt

---

## 1. What this project is
Visual Prompt Tuning adapts a large pre-trained Vision Transformer to a new task **without changing its weights**: the backbone stays frozen and only a few extra input tokens ("prompts") plus a linear head are learned
(*VPT-shallow*: prompts at the first layer only; *VPT-deep*: new prompts at every layer). We ran the **authors' official implementation** (kept byte-identical in `third_party/vpt_official/`) on **Oxford Flowers-102**,
one of the paper's FGVC benchmarks, and compared it with the paper's baselines (Linear, Partial-1, Bias, Adapter, Full fine-tuning), plus a prompt-length ablation and t-SNE analysis.
Our own additions are the thin runner (`official_runner.py`), the experimental protocol, the extra analysis, the checks and this demo (see section 3 for exactly what is ours and what is the authors').

## 2. How to run the project (training, testing, prediction)
The project has three runnable parts. In our case **training and testing run on Kaggle** (Tesla T4 GPU); the prediction demo runs anywhere, even on a CPU-only laptop.

| Part | What it does | Where it runs | Needs | Time in our run |
|---|---|---|---|---|
| **Training + validation + testing** | trains every method with the official trainer, chooses learning rates on validation data, evaluates the test split, creates tables and figures | Kaggle notebook `notebooks/01_train_evaluate_kaggle_executed.ipynb` | GPU (T4), Internet On | about 9.6 GPU-hours (one 9–10 h session) |
| **Prediction demo** (no training) | loads the trained model and classifies any image | PC: `demo.py` · Kaggle/Colab: `demo.ipynb` | CPU is enough; Internet once for the 413 MB weights | minutes |
| **Self-checks** | verifies this package and the demo | PC (`tools/`, `tests/`) | Python | seconds to minutes |

### 2.1 Before you start (training on Kaggle)
* A Kaggle account with a **verified phone number** (required for GPU and Internet) and enough weekly GPU quota — our full run used about 10 of the 30 weekly GPU-hours.
* Nothing else to upload: the training notebook is **self-contained** (the official repository is embedded, the dataset and the pre-trained weights are downloaded by the notebook).

### 2.2 Training and testing on Kaggle — step by step
1. Sign in at kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook** → select `notebooks/01_train_evaluate_kaggle_executed.ipynb` (its old outputs are replaced as you run it).
3. Right panel → **Session options**: **Accelerator = GPU T4** (T4 ×2 is fine; one GPU is used) and **Internet = On**.
4. Press **Run All** and keep the tab open for the whole run — or use **Save Version → Save & Run All (Commit)** to run in the background.
5. Wait. The notebook then does everything by itself, in this order:
   1. unpacks the embedded official repository and checks its SHA-256;
   2. downloads the official ViT-B/16 ImageNet-21k weights (about 413 MB) and the Oxford Flowers-102 dataset, writes the train/validation/test annotation files and checks that the three splits do not overlap;
   3. runs 16 checks of the official model against statements of the paper (parameter counts, prompt initialisation, prompts at every layer, frozen backbone, …);
   4. **trains** 7 methods (Linear, VPT-shallow, VPT-deep, Full fine-tuning, Partial-1, Bias, Adapter) for 30 epochs each over a small learning-rate grid (18 runs), evaluating on the **validation** split after every epoch;
   5. **tests**: after the final epoch of every run it evaluates once on the 6,149-image **test** split (official `Trainer.eval_classifier`);
   6. selects each method's learning rate from validation accuracy only and builds the results table and the check of the paper's claims;
   7. runs the prompt-length ablation (5 more runs), the t-SNE plot, the demo figure and the storage comparison;
   8. verifies that none of the 88 official files was modified and packs everything into `vpt_official_outputs.zip`.
6. **Download the results before closing the session:** refresh the right-hand file panel, open `/kaggle/working`, click ⋮ next to `vpt_official_outputs.zip` → **Download**; also **File → Download Notebook**. Then stop the session (power button). Files of an interactive session are deleted when it ends.

If the session is interrupted, press **Run All** again: every finished run is stored in `results/*.json` and is skipped.

### 2.3 Settings you can change (configuration cell, section 3 of the training notebook)
| Setting | Default | Effect |
|---|---|---|
| `PROFILE` | `"standard"` | 30 epochs and 2–3 learning rates per method; `"fast"` = 15 epochs and fewer learning rates (much shorter; not run by us) |
| `EXTRA_BASELINES` | `True` | `False` skips Partial-1, Bias and Adapter (saves about 2.2 h) |
| `RUN_ABLATION`, `ABLATION_LENGTHS` | `True`, `[1, 5, 50, 100, 200]` | `False` skips the prompt-length ablation (saves about 2.8 h) |
| `SEEDS` | `[0]` | `[0, 1, 2]` reports mean ± std (three times the time) |
| `NUM_PROMPTS` | `10` | prompt length of the VPT runs |
| `BS` | `32` | use `16` if the GPU runs out of memory (the learning rate is rescaled automatically by batch/256) |

### 2.4 How testing works and where to see the results
* The **test split** (6,149 images) is never used for training or for choosing learning rates. It is evaluated **once per run**, after the final epoch, with the official `Trainer.eval_classifier`. The number is stored as `test_top1` in `results/runs_json/<run>.json` and shown in `results/final_results.csv`, `results/summary.md` and section 4 below. The official training log of every run is in `results/logs/<run>/logs.txt`.
* **Re-check this package** (PC, after `pip install -r requirements.txt`): `python tools/verify_package.py` (required files present, official code unmodified, README numbers = result files = checkpoint manifest, executed notebook without errors), `python tests/test_demo_smoke.py` (every shipped checkpoint loads and predicts; needs the weights, see 2.5) and `python tests/test_windows_path_shim.py`.
* A quick look at the trained model on unseen data: `demo.ipynb`, section 6, predicts 8 random images of the official test split. A standalone script that re-scores the saved checkpoints on the whole test split is not included — the full test evaluation is part of the training notebook.

### 2.5 Prediction demo (no training)
Tested on Linux; the folder, image, figure and smoke-test commands were also run on a Windows PC (Python 3.9).
```bash
pip install -r requirements.txt
python demo.py --download-weights            # one time: official ViT-B/16 ImageNet-21k weights (about 413 MB)
python demo.py --image my_flower.jpg         # VPT-deep, top-5 predictions with probabilities
```
Other uses:
```bash
python demo.py --folder sir_images                       # every image in a folder
python demo.py --image a.jpg b.jpg --save-figure out.png # also save a picture with probability bars
python demo.py --image a.jpg --method vpt_shallow        # linear | bias | adapter | vpt_shallow | vpt_deep
python demo.py --image a.jpg --compare                   # all five trained models side by side
python demo.py --list-methods
```
**On Kaggle or Colab:** import `demo.ipynb`, set **Internet = On** (no GPU needed), **Run All**. Without images of your own it predicts random test images; to classify the examiner's images use **Add Input → Upload**, select the image files, create the dataset and run the "examiner's images" cell again. It only **loads the trained model and predicts — no training**.

**Important for interpretation:** the model was trained on **102 flower categories** only. Given any other picture it still answers with the closest flower, and a **high probability does not prove the picture is a flower** (softmax confidence is not an out-of-domain detector). The demo only prints a note when the top-1 probability is below 50 %. Use real flower photos.

**What is saved:** `checkpoints/` holds the task-specific parameters trained in our run (VPT-deep prompts + head = 0.68 MB; see `checkpoints/manifest.json`). The frozen ViT-B/16 backbone is the official public checkpoint (about 413 MB) and is downloaded by `--download-weights`.
Full fine-tuning and Partial-1 checkpoints are not shipped (their trained weights are large); their results are in `results/`.

## 3. What is ours and what is the authors' — and what we found
We do **not** claim the VPT method or the official code as our own. The project is a reproduction, extension and verification of the authors' work:

| | The authors (paper + official repository) | Our team |
|---|---|---|
| Method and code | invented VPT; wrote the model, trainer and configs; released the pre-trained weights | studied the paper; ran the official code **unmodified** (hash-verified) |
| Experiments | 24 tasks, three runs each | Oxford Flowers-102 with a leakage-controlled protocol (learning rate chosen on validation data only, test split evaluated once); 23 training runs on a Kaggle T4 |
| Software compatibility | 2022 code for Python 3.7 / torch 1.7 | run-time shims (three for the old libraries + one for Windows paths) so the official code runs on today's Kaggle and on Windows — **no official file edited** |
| Extra analysis | figures and tables of the paper | Partial-1, Bias and Adapter baselines through the official switch, prompt-length ablation up to 200 prompts, t-SNE of the final features, storage comparison |
| Verification | — | 16 checks of the official model against the paper's own statements, integrity proof of the 88 official files, every saved result re-checked against its training log, package self-check, smoke tests |
| Demo and documentation | training scripts | prediction demo for any image (command line and notebook), tests, this README |

**What the paper reports versus what we measured** (Oxford Flowers-102, one seed; the paper's numbers are averages over five FGVC datasets and three runs, so only the *direction* is comparable):

| Paper statement | Our measurement | Verdict |
|---|---|---|
| VPT-deep beats full fine-tuning (Table 1, FGVC: 89.11 vs 88.54) | 98.62 vs 97.51 (+1.11) with 0.198 % of the parameters | agrees |
| VPT-deep beats the other parameter-efficient methods (Table 1) | beats Linear (97.95), Partial-1 (97.95), Adapter (98.36) and VPT-shallow (98.03); **Bias scored 98.99, 0.37 points higher** | partly — Bias is also the paper's closest rival (88.41 vs 89.11) |
| VPT-deep is better than VPT-shallow (89.11 vs 84.62) | 98.62 vs 98.03 | agrees (smaller gap on this nearly saturated dataset) |
| 50 prompts add 0.04 % (shallow) and 0.53 % (deep) of the parameters | confirmed on the official model (16/16 checks) | agrees |
| The best prompt length depends on the task (Fig. 6) | best at 5 prompts (98.76); 200 prompts drop to 94.67 (learning rate was kept from 10 prompts) | agrees |

## 4. Results (Oxford Flowers-102; test split of 6,149 images; one seed)
| Method | Test top-1 (%) | Tuned parameters (% of backbone, incl. head) | Learning rate (base) | Checkpoint in `checkpoints/` |
|---|---|---|---|---|
| Linear | 97.95 | 0.091 | 25 | yes |
| Partial-1 | 97.95 | 8.347 | 1 | no (large / not kept) |
| Bias | 98.99 | 0.211 | 1 | yes |
| Adapter | 98.36 | 2.120 | 1 | yes |
| VPT-shallow | 98.03 | 0.100 | 5 | yes |
| VPT-deep | 98.62 | 0.198 | 25 | yes |
| Full fine-tuning | 97.51 | 100.000 | 0.001 | no (large / not kept) |

* **VPT-deep beats full fine-tuning by 1.11 points** while tuning only about 0.2 % of the parameters, and beats VPT-shallow and Linear — the main claims of the paper.
* **Honest note:** Bias scored 0.37 points *higher* than VPT-deep on this dataset (about 23 test images; not conclusive with one seed). The paper's "outperforms all" statement is an average over five datasets, where Bias is also its closest rival.
* Numbers are not comparable with the paper's tables, which average five FGVC datasets and three runs. Plots: `results/figures/` (accuracy overview, training curves, ablation, t-SNE, demo predictions).

**Prompt-length ablation (VPT-deep, learning rate fixed from p = 10, paper Fig. 6):**

| Prompts per layer p | Test top-1 (%) | Tuned parameters (%) |
|---|---|---|
| 1 | 98.05 | 0.102 |
| 5 | 98.76 | 0.145 |
| 10 | 98.62 | 0.198 |
| 50 | 97.85 | 0.625 |
| 100 | 97.66 | 1.152 |
| 200 | 94.67 | 2.191 |

## 5. How the results were produced (protocol and verification)
* **Data and splits:** Oxford Flowers-102 official splits — train 1,020 / validation 1,020 / test 6,149; checked for zero overlap. The learning rate of every method was chosen on **validation** accuracy only; the test split was evaluated **once** per run (final epoch).
* **Training:** the official trainer, loaders and configs (`configs/{linear,prompt,finetune}/flowers.yaml`), ViT-B/16 ImageNet-21k, batch 32 (learning rate scaled by batch/256, the official rule), 30 epochs with 3 warm-up epochs, weight decay 1e-4, prompt length 10, SGD (momentum 0.9) for Linear / prompt / Partial-1 / Bias / Adapter and AdamW for full fine-tuning (official defaults). One Tesla T4 (Kaggle); the full-fine-tuning runs took about 29 minutes each.
* **Evidence of correctness:** 16/16 checks of the official model against the paper's own statements (0.04 % / 0.53 % prompt parameters for 50 prompts, prompt initialisation, fresh prompts at every layer, location invariance, frozen backbone) — `results/paper_conformance.json`; all 88 official files unchanged after the run — `results/official_code_integrity.json`;
  every saved result re-checked against its own official training log (23 runs, 0 mismatches); `tools/verify_package.py` re-checks this package.
* **Deviations from the paper (runtime):** one dataset instead of 24 tasks, 30 instead of 100 epochs, a 3-value learning-rate grid, one seed instead of three, prompt length fixed to 10 (plus ablation), Mlp-3 and Sidetune baselines not run.

## 6. What is in this zip — and what you need for each way of running
| File / folder | What it is | Needed for |
|---|---|---|
| `demo.ipynb` | **Self-contained demo notebook**: the whole project is embedded, so *import it into Kaggle/Colab and Run All* — no other file is required (Internet On for the weights) | notebook demo |
| `demo.py` | command-line demo (`python demo.py --image ...`) | PC demo |
| `official_runner.py` | thin glue that calls the official classes; also holds the compatibility shims (including the Windows path fix) | `demo.py`, `demo.ipynb` |
| `flower_names.py` | names of the 102 classes (index → flower name) | `demo.py`, `demo.ipynb` |
| `checkpoints/` | trained prompts / bias / adapters + heads (the "model" we trained) and `manifest.json` | every demo |
| `third_party/vpt_official/` | the authors' official repository, **unmodified** (`MANIFEST.sha256` proves it) | every demo |
| `requirements.txt` | Python packages for the PC demo | PC demo |
| `results/` | final tables, 5 figures, per-run results, official training logs, conformance and integrity evidence | report / PPT / viva |
| `notebooks/01_train_evaluate_kaggle_executed.ipynb` | the full training + evaluation notebook with all outputs (self-contained, produced the results) | reproduce training |
| `tests/`, `tools/verify_package.py` | smoke tests and a consistency check of this package | quality assurance |
| `README.md`, `NOTICE.md`, `sir_images/` | this file, attribution/license, folder for the examiner's images | — |

**Three ways to use it**
* **A — Kaggle or Colab (nothing else needed):** import `demo.ipynb`, Internet On, *Run All*. Without images it demonstrates the model on random test images; to use the examiner's images attach them as a dataset (Add Input → Upload) and run the "examiner's images" cell again.
* **B — PC (tested on Windows):** unzip, `python -m pip install -r requirements.txt`, `python demo.py --download-weights`, then `python demo.py --image photo.jpg`. Everything needed except the downloaded weights is in the zip.
* **C — Train and test (reproduce the results):** run `notebooks/01_train_evaluate_kaggle_executed.ipynb` on Kaggle with a GPU — step by step in section 2.2.
Why the `.py` files exist: the notebooks are self-contained, but the `.py` files let the demo run on a PC without Jupyter or Kaggle, and they are the readable source of the demo logic.

**Windows note:** the official code builds checkpoint keys with `os.path.join`, which breaks on Windows. `official_runner.py` fixes this at run time (tested by `tests/test_windows_path_shim.py`); do **not** edit files inside `third_party/`.

## 7. Troubleshooting
* **Training on Kaggle:** *no GPU / Internet option* → verify your phone number in Kaggle settings; *the run stops or the page disconnects* → press Run All again (finished runs are skipped); *"CUDA out of memory"* → set `BS = 16` in the configuration cell; *the zip is not visible in the file panel* → click the refresh icon of the panel; *files vanished* → an interactive session deletes its files when it ends, so download first.
* **Demo:** *"weights missing"* → `python demo.py --download-weights` (needs internet once). If it fails, download the file from the URL printed in the error and save it as `weights/imagenet21k_ViT-B_16.npz`. *No GPU* → fine; inference runs on CPU. *Strange predictions* → the image is probably not one of the 102 flowers (see the confidence note). *Unreadable image* → the demo skips it with a warning.
* Check the package: `python tools/verify_package.py`; test the demo: `python tests/test_demo_smoke.py` (after the weights are downloaded).

## 8. Attribution, license, citation
The official code (`third_party/vpt_official/`) is by the authors of the paper under **CC BY-NC 4.0** (see its `LICENSE`) and is redistributed unmodified for non-commercial, academic use. Our added files are provided for coursework evaluation. See `NOTICE.md`.
```bibtex
@inproceedings{jia2022vpt,
  title     = {Visual Prompt Tuning},
  author    = {Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
