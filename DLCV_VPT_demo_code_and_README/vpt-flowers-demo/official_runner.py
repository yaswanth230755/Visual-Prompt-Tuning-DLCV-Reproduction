"""
Thin glue around the UNMODIFIED official VPT code (github.com/KMnP/vpt, Jia et al., ECCV 2022).

Nothing inside the official repository is edited. We only
  (1) add three compatibility shims so the 2022 code runs on a modern Python/timm
      (`turtle` needs tkinter; `timm.models.layers.helpers` was removed from timm;
       `json.load(..., encoding=)` was removed in Python 3.9),
  (2) call the official classes in exactly the order of the official train.py:
      get_cfg -> data loaders -> build_model -> Trainer.train_classifier -> Trainer.eval_classifier.
"""
import os
import random
import re
import sys
import time
import types

import numpy as np
import torch

METHOD_CONFIG = {                       # official config file (configs/<dir>/flowers.yaml) per method
    "linear": ("linear", []),
    "vpt_shallow": ("prompt", ["MODEL.PROMPT.DEEP", "False"]),
    "vpt_deep": ("prompt", ["MODEL.PROMPT.DEEP", "True"]),
    "full": ("finetune", []),           # TRANSFER_TYPE "end2end" = full fine-tuning
    # extra baselines of the paper's Table 1 (all implemented by the official repo via MODEL.TRANSFER_TYPE)
    "partial1": ("linear", ["MODEL.TRANSFER_TYPE", "partial-1"]),    # fine-tune the last block + head
    "bias": ("linear", ["MODEL.TRANSFER_TYPE", "tinytl-bias"]),      # bias terms + head (BitFit / TinyTL)
    "adapter": ("linear", ["MODEL.TRANSFER_TYPE", "adapter"]),       # Houlsby/Pfeiffer adapters + head
}


def prepare(repo_dir):
    """Make the official package importable (shims only; no file of the repo is changed)."""
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    t = types.ModuleType("turtle")
    t.forward = lambda *a, **k: None
    sys.modules.setdefault("turtle", t)
    os.environ.setdefault("SLURMD_NODENAME", "localhost")     # read by the official setup(); value unused here
    try:                                   # timm >= 0.9 removed timm.models.layers.helpers (old timm still has it -> no shim needed)
        import timm.layers as L
        h = types.ModuleType("timm.models.layers.helpers")
        h.to_2tuple = L.to_2tuple
        sys.modules.setdefault("timm.models.layers.helpers", h)
    except ImportError:
        pass
    import functools
    import json as _json
    if not getattr(_json.load, "_vpt_shim", False):          # official io_utils calls json.load(..., encoding=...) (removed in Python 3.9)
        _orig = _json.load

        @functools.wraps(_orig)
        def _load(fp, *a, encoding=None, **k):
            return _orig(fp, *a, **k)
        _load._vpt_shim = True
        _json.load = _load
    _import_vit_with_posix_join()
    return repo_dir


def _import_vit_with_posix_join():
    # Windows fix: the official vit.py builds the keys of the .npz checkpoint with `from os.path import join as pjoin`.
    # npz keys use '/', but on Windows os.path.join inserts a backslash -> KeyError. We import that module once with
    # posixpath.join bound as pjoin (then restore os.path.join), so NO official file has to be edited.
    import importlib
    import posixpath
    _orig_join = os.path.join
    os.path.join = posixpath.join
    try:
        importlib.import_module("src.models.vit_backbones.vit")
    finally:
        os.path.join = _orig_join


def make_cfg(repo_dir, method, lr, wd, epochs, warmup, batch_size, data_path, model_root, out_dir,
             num_tokens=10, workers=2, feature="sup_vitb16_imagenet21k", num_classes=102, seed=0, num_gpus=None):
    from src.configs.config import get_cfg
    sub, extra = METHOD_CONFIG[method]
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(repo_dir, "configs", sub, "flowers.yaml"))
    opts = [
        "DATA.DATAPATH", data_path, "DATA.FEATURE", feature, "DATA.NUMBER_CLASSES", str(num_classes),
        "DATA.BATCH_SIZE", str(batch_size), "DATA.NUM_WORKERS", str(workers), "DATA.NO_TEST", "True",
        "MODEL.MODEL_ROOT", model_root, "MODEL.PROMPT.NUM_TOKENS", str(num_tokens),
        "SOLVER.BASE_LR", str(lr / 256 * batch_size),         # same lr scaling rule as official tune_fgvc.py
        "SOLVER.WEIGHT_DECAY", str(wd), "SOLVER.TOTAL_EPOCH", str(epochs), "SOLVER.WARMUP_EPOCH", str(warmup),
        "SOLVER.LOG_EVERY_N", "50", "OUTPUT_DIR", out_dir, "SEED", str(seed),
        "NUM_GPUS", str(num_gpus if num_gpus is not None else 1),       # official loader divides by NUM_GPUS
    ] + extra
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def _parse_train_loss(log_path):
    out = {}
    if os.path.exists(log_path):
        for m in re.finditer(r"Epoch (\d+) / \d+: .*?average train loss: ([0-9.]+)", open(log_path).read()):
            out[int(m.group(1))] = float(m.group(2))
    return [out[k] for k in sorted(out)]


def run_official(cfg, save_state=True, tsne_n=0):
    """Train one configuration with the official Trainer; evaluate the test split once at the end."""
    from src.data import loader as data_loader
    from src.engine.evaluator import Evaluator
    from src.engine.trainer import Trainer
    from src.models.build_model import build_model
    import src.utils.logging as logging

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logging.setup_logging(cfg.NUM_GPUS, 1, cfg.OUTPUT_DIR, name="visual_prompt")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(0)     # as in official train.py

    t0 = time.time()
    train_loader = data_loader.construct_train_loader(cfg)
    val_loader = data_loader.construct_val_loader(cfg)
    test_loader = data_loader.construct_test_loader(cfg)
    model, device = build_model(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    evaluator = Evaluator()
    trainer = Trainer(cfg, model, evaluator, device)
    trainer.train_classifier(train_loader, val_loader, None)       # official loop (val each epoch)
    total = cfg.SOLVER.TOTAL_EPOCH
    res = evaluator.results
    name = cfg.DATA.NAME
    val_curve = [float(res[f"epoch_{e}"]["classification"][f"val_{name}"]["top1"]) for e in range(total)
                 if f"epoch_{e}" in res and f"val_{name}" in res[f"epoch_{e}"].get("classification", {})]
    diverged = len(val_curve) < total
    out = dict(val_curve=val_curve, diverged=diverged, n_total_params=n_total, n_trainable_params=n_train,
               tuned_percent=100.0 * n_train / n_total)
    if not diverged:
        trainer.eval_classifier(test_loader, "test", True)           # official evaluation, once, final epoch
        out["test_top1"] = float(res[f"epoch_{total - 1}"]["classification"][f"test_{name}"]["top1"])
        out["test_top5"] = float(res[f"epoch_{total - 1}"]["classification"][f"test_{name}"].get("top5", float("nan")))
    out["val_top1_final"] = val_curve[-1] if val_curve else float("nan")
    out["val_top1_best"] = max(val_curve) if val_curve else float("nan")
    out["train_loss_curve"] = _parse_train_loss(os.path.join(cfg.OUTPUT_DIR, "logs.txt"))
    if tsne_n and not diverged:                                   # features of a FIXED test subset (paper Fig. 9), same images for every method
        ds = official_dataset(cfg, "test")
        sel = np.random.default_rng(0).choice(len(ds), size=min(tsne_n, len(ds)), replace=False)
        model.eval()
        feats, labs = [], []
        with torch.no_grad():
            for i in range(0, len(sel), 32):
                items = [ds[int(j)] for j in sel[i:i + 32]]
                x = torch.stack([it["image"] for it in items]).to(device)
                feats.append(model.get_features(x).float().cpu())
                labs += [int(it["label"]) for it in items]
        np.savez_compressed(os.path.join(cfg.OUTPUT_DIR, "tsne_features.npz"), feats=torch.cat(feats).numpy(), labels=np.array(labs))
    out["seconds"] = time.time() - t0
    if save_state and not diverged and cfg.MODEL.TRANSFER_TYPE != "end2end":
        keep = {n for n, p in model.named_parameters() if p.requires_grad}
        torch.save({k: v.cpu() for k, v in model.state_dict().items() if k in keep},
                   os.path.join(cfg.OUTPUT_DIR, "trainable_state.pth"))
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# inference helpers (use the official model class, the official test transform)
# --------------------------------------------------------------------------- #
def load_trained(cfg, state_path=None):
    """Build the official model for `cfg` (loads the pretrained ViT) and load the saved prompts/head."""
    from src.models.build_model import build_model
    model, device = build_model(cfg)
    if state_path:
        sd = torch.load(state_path, map_location="cpu")
        miss = model.load_state_dict(sd, strict=False)
        assert not miss.unexpected_keys, miss.unexpected_keys
    return model.eval(), device


@torch.no_grad()
def features_and_logits(model, device, images):
    """images: normalised float tensor (B,3,224,224) produced by the official test transform."""
    x = images.to(device)
    f = model.get_features(x)
    return model.head(f).float().cpu(), f.float().cpu()


def official_test_transform(size=224):
    from src.data.transforms import get_transforms
    return get_transforms("test", size)


def official_dataset(cfg, split):
    from src.data.datasets.json_dataset import FlowersDataset
    return FlowersDataset(cfg, split)


# --------------------------------------------------------------------------- #
# smoke-test helpers (only used when the notebook is run with VPT_SMOKE=1 on a CPU box)
# --------------------------------------------------------------------------- #
def make_fake_flowers(root, n_classes=10):
    """Tiny dataset in the official FGVC layout (<split>.json + images) for pipeline tests."""
    from PIL import Image
    import json
    rng = np.random.default_rng(0)
    os.makedirs(root, exist_ok=True)
    for split, n in (("train", 4), ("val", 1), ("test", 2)):
        ann = {}
        for c in range(n_classes):
            for i in range(n):
                fn = f"{split}_{c}_{i}.jpg"
                Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(os.path.join(root, fn))
                ann[fn] = c
        json.dump(ann, open(os.path.join(root, f"{split}.json"), "w"))


def make_fake_weights(repo_dir, out_path):
    """Random ViT-B/16 weights in the Google .npz layout expected by the official loader."""
    from src.models.vit_backbones.vit import VisionTransformer
    m = VisionTransformer("sup_vitb16_imagenet21k", 224, num_classes=-1)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    w = {"embedding/kernel": sd["transformer.embeddings.patch_embeddings.weight"].transpose(2, 3, 1, 0),
         "embedding/bias": sd["transformer.embeddings.patch_embeddings.bias"],
         "cls": sd["transformer.embeddings.cls_token"],
         "Transformer/encoder_norm/scale": sd["transformer.encoder.encoder_norm.weight"],
         "Transformer/encoder_norm/bias": sd["transformer.encoder.encoder_norm.bias"],
         "Transformer/posembed_input/pos_embedding": sd["transformer.embeddings.position_embeddings"]}
    A = "MultiHeadDotProductAttention_1/"
    for i in range(12):
        R_, o = f"Transformer/encoderblock_{i}/", f"transformer.encoder.layer.{i}."
        for nm in ("query", "key", "value"):
            w[R_ + A + nm + "/kernel"] = sd[o + f"attn.{nm}.weight"].T.reshape(768, 12, 64)
            w[R_ + A + nm + "/bias"] = sd[o + f"attn.{nm}.bias"].reshape(12, 64)
        w[R_ + A + "out/kernel"] = sd[o + "attn.out.weight"].T.reshape(12, 64, 768)
        w[R_ + A + "out/bias"] = sd[o + "attn.out.bias"]
        for a, b in (("Dense_0", "fc1"), ("Dense_1", "fc2")):
            w[R_ + f"MlpBlock_3/{a}/kernel"] = sd[o + f"ffn.{b}.weight"].T
            w[R_ + f"MlpBlock_3/{a}/bias"] = sd[o + f"ffn.{b}.bias"]
        for a, b in (("LayerNorm_0", "attention_norm"), ("LayerNorm_2", "ffn_norm")):
            w[R_ + a + "/scale"] = sd[o + b + ".weight"]
            w[R_ + a + "/bias"] = sd[o + b + ".bias"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **w)


# --------------------------------------------------------------------------- #
# verification helpers
# --------------------------------------------------------------------------- #
def paper_conformance(repo_dir, model_root=None, num_classes=102, verbose=True):
    """
    Checks the OFFICIAL implementation against statements of the paper (Sec. 3, 'Storing Visual Prompts').
    Returns a list of (claim, passed, detail). Never raises (a failing check is reported, not fatal).
    """
    import math
    from types import SimpleNamespace
    from src.models.vit_prompt.vit import PromptedVisionTransformer
    out = []

    def check(name, ok, info=""):
        out.append((name, bool(ok), info))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}")

    p, d, L, m_tokens = 50, 768, 12, 196
    for deep in (False, True):
        tag = "VPT-deep" if deep else "VPT-shallow"
        cfg = SimpleNamespace(LOCATION="prepend", INITIATION="random", NUM_DEEP_LAYERS=None, DEEP_SHARED=False,
                              NUM_TOKENS=p, DROPOUT=0.0, PROJECT=-1, DEEP=deep, VIT_POOL_TYPE="original")
        torch.manual_seed(0)
        net = PromptedVisionTransformer(cfg, "sup_vitb16_imagenet21k", img_size=224, num_classes=num_classes).eval()
        backbone = sum(q.numel() for n, q in net.named_parameters() if "prompt" not in n and not n.startswith("head"))
        extra = net.transformer.prompt_embeddings.numel() + (net.transformer.deep_prompt_embeddings.numel() if deep else 0)
        want = L * p * d if deep else p * d
        pct = 100.0 * extra / backbone
        check(f"{tag}: ViT-B backbone has ~86 M parameters", abs(backbone / 1e6 - 85.8) < 0.3, f"{backbone / 1e6:.2f} M")
        check(f"{tag}: p=50 prompts add {want:,} parameters ({'N*p*d' if deep else 'p*d'})", extra == want, f"{extra:,}")
        check(f"{tag}: = {0.53 if deep else 0.04}% of the backbone (paper Sec. 3.2)", abs(pct - (0.53 if deep else 0.04)) < (0.02 if deep else 0.01), f"{pct:.3f}%")
        bound = math.sqrt(6.0 / (3 * 16 * 16 + d))
        mx = max(net.transformer.prompt_embeddings.abs().max().item(),
                 net.transformer.deep_prompt_embeddings.abs().max().item() if deep else 0)
        check(f"{tag}: prompts initialised U(-v, v), v = sqrt(6/(3*16^2+d))", mx <= bound + 1e-6, f"max|p|={mx:.4f} <= {bound:.4f}")
        seen = {}
        hooks = [blk.register_forward_pre_hook(lambda mod, inp, i=i: seen.__setitem__(i, inp[0].detach()))
                 for i, blk in enumerate(net.transformer.encoder.layer)]
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            net(x)
        [h.remove() for h in hooks]
        check(f"{tag}: every layer sees [CLS] + {p} prompts + {m_tokens} patches", all(v.shape[1] == 1 + p + m_tokens for v in seen.values()),
              f"seq len {seen[0].shape[1]}")
        first = torch.allclose(seen[0][:, 1:1 + p], net.transformer.prompt_embeddings.expand(2, -1, -1))
        if deep:
            later = all(torch.allclose(seen[i][:, 1:1 + p], net.transformer.deep_prompt_embeddings[i - 1].expand(2, -1, -1)) for i in range(1, L))
            check("VPT-deep: layer i receives its own fresh prompts P_(i-1) (Eq. 7)", first and later)
        else:
            check("VPT-shallow: prompts enter layer 1 only (Eq. 4-6)", first)
            with torch.no_grad():                                    # location invariance claimed in Sec. 3.2
                ref = net.transformer(x)[0][:, 0]
                z = torch.cat([net.transformer.embeddings(x), net.transformer.prompt_embeddings.expand(2, -1, -1)], 1)   # [x0, E0, P]
                for blk in net.transformer.encoder.layer:
                    z, _ = blk(z)
                alt = net.transformer.encoder.encoder_norm(z)[:, 0]
            check("[x0,P,E0] and [x0,E0,P] give the same [CLS] output (paper Sec. 3.2)", torch.allclose(ref, alt, atol=1e-4),
                  f"max|diff|={(ref - alt).abs().max().item():.1e}")
        del net
    if model_root is not None:                                      # frozen-backbone check on the real build_model path
        from src.models.build_model import build_model
        for method, want_names in (("vpt_deep", {"prompt_embeddings", "deep_prompt_embeddings", "head"}), ("vpt_shallow", {"prompt_embeddings", "head"}), ("linear", {"head"})):
            cfg = make_cfg(repo_dir, method, 1.0, 1e-4, 1, 1, 2, ".", model_root, "conformance_tmp", num_tokens=p, workers=0, num_classes=num_classes)
            net, _ = build_model(cfg)
            names = {n for n, q in net.named_parameters() if q.requires_grad}
            kinds = {("deep_prompt_embeddings" if "deep_prompt_embeddings" in n else "prompt_embeddings" if "prompt_embeddings" in n else "head" if "head" in n else n) for n in names}
            check(f"{method}: only {sorted(want_names)} are trainable (backbone frozen)", kinds == want_names, f"{len(names)} tensors")
            del net
    return out


def file_manifest(root, skip=("__pycache__",)):
    """sha256 of every file below `root` (used to prove the official sources were not modified)."""
    import hashlib
    man = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            if f.endswith(".pyc"):
                continue
            path = os.path.join(dp, f)
            man[os.path.relpath(path, root)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return man
