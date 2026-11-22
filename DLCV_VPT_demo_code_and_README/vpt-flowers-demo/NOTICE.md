# Notice - third-party code and data

* `third_party/vpt_official/` is the repository https://github.com/KMnP/vpt by Menglin Jia et al., licensed CC BY-NC 4.0 (`third_party/vpt_official/LICENSE`).
  It is included **unmodified** (SHA-256 of the supplied archive: `9c6839289c34960eeb3d19d6950daa67604ab5e0e94011bc38b70389660c1ce1`; per-file hashes in `MANIFEST.sha256`). Compatibility with modern Python/timm is achieved at run time by three shims in `official_runner.py`, never by editing these files.
* Pre-trained ViT-B/16 (ImageNet-21k) weights are the public checkpoint released with the paper's code; they are downloaded, not redistributed here.
* Dataset: Oxford Flowers-102 (Nilsback & Zisserman, 2008), obtained through torchvision; not redistributed here.
* Files written by the team: `demo.py`, `demo.ipynb`, `official_runner.py`, `flower_names.py`, `tests/`, `tools/`, `README.md`, the training notebook and the analysis in `results/`.
