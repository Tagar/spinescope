"""Train, export and evaluate the learned spine labeller (``spinescope.anatomy.ml``). Dev-time only.

Needs the ``train`` extra (PyTorch, ONNX) and the ``volumes`` extra (SimpleITK), e.g.::

    uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev,train]"

Data: the SPIDER lumbar MRI dataset (van der Graaf et al., Scientific Data 2024,
doi:10.1038/s41597-024-03090-w; CC BY 4.0), read from ``$SPINESCOPE_DATA/spider`` (default
``~/data/public/spider``). Only the ``training`` subset of ``overview.csv`` is used for fitting; the
``validation`` subset is only read by ``evaluate``. Nothing from SPIDER is written into the repository:
the cache and checkpoints go to ``--out`` (default ``/tmp/spinescope-anatomy_ml``).

Subcommands::

    train     prepare the training slices, fit the U-Net, export ONNX (+ metadata JSON)
    evaluate  run label_spine_ml on a subset and report anatomy and segmentation metrics (JSON)

Reproducibility: ``--seed`` seeds Python, NumPy and PyTorch. The data order, augmentation and
initialisation are then fixed; Apple MPS kernels are not bit-exact across machines, so reruns agree
to within training noise. The versions used are recorded in the metadata file.

Slices: sagittal slices resampled to a 1 mm grid (rows superior->inferior, columns anterior->posterior,
see ``spinescope.anatomy.ml.grid``), on a 320 x 320 canvas centred on the spine, from the slices within
``--para-mm`` of the canal centre (midline plus parasagittal). Augmentation: small rotation, scale and
translation, gamma, brightness and contrast, a smooth multiplicative bias field and noise. No flips
(A/P orientation is clinically meaningful). Loss: cross-entropy (class weights) + soft Dice on the
foreground classes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from spinescope.anatomy.ml.grid import intensity_window, normalise, plane_grid, resample_slice
from spinescope.config import DEFAULT

log = logging.getLogger("train_labeller")

CLASS_CANAL, CLASS_BODY, CLASS_DISC = 3, 1, 2
CLASS_NAMES = {1: "vertebral body", 2: "disc", 3: "spinal canal"}
N_CLASSES = 4
CANVAS = 320  # training canvas (1 mm pixels) centred on the spine
PATCH = 256  # network training patch (random affine crop of the canvas)
WIDTHS = (16, 32, 64, 128, 128)
CLASS_WEIGHTS = (0.5, 1.0, 2.0, 2.0)  # background, body, disc, canal


def spider_root(arg: str | None) -> Path:
    base = Path(arg) if arg else Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public"))
    return base / "spider" if base.name != "spider" else base


def rows_for(root: Path, subset: str) -> list[dict[str, str]]:
    with open(root / "overview.csv", newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["subset"] == subset]


def to_canvas(img: np.ndarray, lab: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre a (H, W) slice and its labels on a size x size canvas (pad with background, crop if larger)."""
    h, w = img.shape
    fg = np.argwhere(lab > 0)
    cy, cx = fg.mean(axis=0) if len(fg) else (h / 2, w / 2)
    y0 = int(np.rint(cy)) - size // 2
    x0 = int(np.rint(cx)) - size // 2
    out_i = np.zeros((size, size), np.float32)
    out_l = np.zeros((size, size), np.uint8)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + size), min(w, x0 + size)
    if ye > ys and xe > xs:
        out_i[ys - y0 : ye - y0, xs - x0 : xe - x0] = img[ys:ye, xs:xe]
        out_l[ys - y0 : ye - y0, xs - x0 : xe - x0] = lab[ys:ye, xs:xe]
    return out_i, out_l


def class_map(labels: np.ndarray) -> np.ndarray:
    """SPIDER label volume -> classes: 0 background, 1 vertebral body (1-99), 2 disc (200-299), 3 canal (100)."""
    v = np.rint(labels).astype(np.int32)
    cls = np.zeros(v.shape, np.uint8)
    cls[(v >= 1) & (v <= 99)] = CLASS_BODY
    cls[(v >= 200) & (v <= 299)] = CLASS_DISC
    cls[v == 100] = CLASS_CANAL
    return cls


def prepare_volume(name: str, root: Path, para_mm: float, res: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Canvas images (float32 in [0,1]) and class labels for the slices of one SPIDER volume."""
    from spinescope.io.volumes import read_volume

    st = read_volume(str(root / "images" / f"{name}.mha"))
    mk = read_volume(str(root / "masks" / f"{name}.mha"))
    if st.shape != mk.shape or not np.allclose(st.ipp, mk.ipp, atol=1e-3):
        raise ValueError(f"{name}: image and mask geometry differ")
    if st.plane != "sag":
        log.warning("%s: not a sagittal volume (%s), skipped", name, st.plane)
        return None
    cls = class_map(mk.vol)
    areas = (cls == CLASS_CANAL).sum(axis=(1, 2))
    if areas.sum() == 0:
        log.warning("%s: no canal label, skipped", name)
        return None
    pos = np.asarray(st.positions, dtype=np.float64)
    x_canal = float(np.average(pos, weights=areas))
    window = intensity_window(st.vol, DEFAULT.ml_intensity_pct, DEFAULT.background_frac)
    imgs, labs = [], []
    for k in range(st.n):
        if abs(pos[k] - x_canal) > para_mm:
            continue
        g = plane_grid(st, k, res)
        img = normalise(resample_slice(st, g), window)
        lab = np.rint(resample_slice(st, g, array=cls, order=0)).astype(np.uint8)
        ci, cl = to_canvas(img, lab, CANVAS)
        imgs.append(ci)
        labs.append(cl)
    if not imgs:
        return None
    return np.stack(imgs), np.stack(labs)


def prepare(root: Path, subset: str, para_mm: float, res: float, limit: int | None) -> dict[str, Any]:
    """Cache the slices of a SPIDER subset in memory (float16 images, uint8 labels)."""
    rows = rows_for(root, subset)
    if limit:
        rows = rows[:limit]
    xs, ys, names = [], [], []
    for i, r in enumerate(rows):
        name = r["new_file_name"]
        res_v = prepare_volume(name, root, para_mm, res)
        if res_v is None:
            continue
        xs.append(res_v[0].astype(np.float16))
        ys.append(res_v[1])
        names.extend([name] * len(res_v[0]))
        if (i + 1) % 50 == 0:
            log.info("prepared %d/%d volumes (%d slices)", i + 1, len(rows), sum(len(x) for x in xs))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return {"x": x, "y": y, "names": names, "volumes": len(xs)}


# ----------------------------------------------------------------------------- model and training
def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("training needs PyTorch: uv pip install -e '.[train]'") from exc
    return torch


def build_model(widths: tuple[int, ...] = WIDTHS, n_classes: int = N_CLASSES) -> Any:
    """Small 2-D U-Net: 4 down-sampling stages, batch norm, transposed-conv up-sampling."""
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.downs = nn.ModuleList()
            cin = 1
            for w in widths:
                self.downs.append(block(cin, w))
                cin = w
            self.ups = nn.ModuleList()
            self.decs = nn.ModuleList()
            for w_skip, w_below in reversed(list(itertools.pairwise(widths))):
                self.ups.append(nn.ConvTranspose2d(w_below, w_below, kernel_size=2, stride=2))
                self.decs.append(block(w_below + w_skip, w_skip))
            self.head = nn.Conv2d(widths[0], n_classes, kernel_size=1)

        def forward(self, x: Any) -> Any:
            skips = []
            for i, b in enumerate(self.downs):
                x = b(x)
                if i < len(self.downs) - 1:
                    skips.append(x)
                    x = F.max_pool2d(x, 2)
            for up, dec in zip(self.ups, self.decs, strict=True):
                x = up(x)
                x = dec(torch.cat([x, skips.pop()], dim=1))
            return self.head(x)

    return UNet()


def augment(x: Any, y: Any, out: int) -> tuple[Any, Any]:
    """Random rotation (+-10 deg), scale (+-15%), translation (+-12% of canvas), gamma, contrast,
    brightness, smooth bias field and noise. Labels follow the geometry (nearest sampling)."""
    torch = _torch()
    F = torch.nn.functional
    b = x.shape[0]
    dev = x.device
    rnd = lambda *s: torch.rand(*s, device=dev) * 2 - 1  # noqa: E731  (uniform in [-1, 1))
    ang = rnd(b) * math.radians(10.0)
    sc = 1.0 + rnd(b) * 0.15
    f = out / x.shape[-1]
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = f * sc * cos
    theta[:, 0, 1] = -f * sc * sin
    theta[:, 1, 0] = f * sc * sin
    theta[:, 1, 1] = f * sc * cos
    theta[:, 0, 2] = rnd(b) * 0.12
    theta[:, 1, 2] = rnd(b) * 0.12
    grid = F.affine_grid(theta, (b, 1, out, out), align_corners=False)
    xo = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    yo = F.grid_sample(y[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    yo = yo[:, 0].round().long()
    gamma = torch.exp(rnd(b, 1, 1, 1) * 0.25)
    xo = xo.clamp(0, 1) ** gamma
    xo = xo * (1.0 + rnd(b, 1, 1, 1) * 0.15) + rnd(b, 1, 1, 1) * 0.05
    field = torch.exp(
        F.interpolate(
            torch.randn(b, 1, 4, 4, device=dev) * 0.15, size=(out, out), mode="bicubic", align_corners=False
        )
    )
    xo = xo * field + torch.randn_like(xo) * 0.02 * torch.rand(b, 1, 1, 1, device=dev)
    return xo, yo


def dice_ce_loss(logits: Any, y: Any) -> Any:
    torch = _torch()
    F = torch.nn.functional
    w = torch.tensor(CLASS_WEIGHTS, device=logits.device)
    ce = F.cross_entropy(logits, y, weight=w)
    p = logits.softmax(dim=1)
    oh = F.one_hot(y, N_CLASSES).permute(0, 3, 1, 2).to(p.dtype)
    inter = (p * oh).sum(dim=(0, 2, 3))
    den = p.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (den + 1.0)
    return ce + dice[1:].mean()


def pick_device(torch: Any, name: str) -> Any:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=Path(__file__).parent
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "torch", "onnx", "onnxruntime", "SimpleITK", "spinescope"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except ImportError:
            out[mod] = "missing"
    return out


def cmd_train(args: argparse.Namespace) -> None:
    torch = _torch()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = spider_root(args.spider)
    t0 = time.time()
    cache = out / f"train_cache_{args.para_mm:g}mm_{args.limit or 'all'}.npz"
    if cache.exists() and not args.rebuild:
        d = np.load(cache, allow_pickle=True)
        x, y, n_vol = d["x"], d["y"], int(d["volumes"])
        log.info("loaded cached slices from %s", cache)
    else:
        data = prepare(root, "training", args.para_mm, args.res, args.limit)
        x, y, n_vol = data["x"], data["y"], data["volumes"]
        np.savez(cache, x=x, y=y, volumes=np.array(n_vol))
    log.info("training slices: %d from %d volumes (%.0f s)", len(x), n_vol, time.time() - t0)
    device = pick_device(torch, args.device)
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %d parameters on %s", n_params, device)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    steps_per_epoch = max(1, len(xt) // args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = min(200, total // 10)

    def lr_at(s: int) -> float:
        if s < warm:
            return args.lr * (s + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    step = 0
    history = []
    gen = torch.Generator().manual_seed(seed)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        t_ep = time.time()
        for it in range(steps_per_epoch):
            idx = perm[it * args.batch : (it + 1) * args.batch]
            xb = xt[idx].to(device=device, dtype=torch.float32)[:, None]
            yb = yt[idx].to(device)
            xb, yb = augment(xb, yb, PATCH)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            logits = model(xb)
            loss = dice_ce_loss(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach().cpu())
            step += 1
        mean = tot / steps_per_epoch
        history.append(mean)
        log.info("epoch %d/%d loss %.4f (%.0f s)", ep + 1, args.epochs, mean, time.time() - t_ep)
        torch.save(model.state_dict(), out / "checkpoint.pt")
    export_onnx(model.cpu().eval(), out / "labeller_v1.onnx")
    meta = {
        "model": "labeller_v1",
        "widths": list(WIDTHS),
        "parameters": n_params,
        "classes": ["background", "vertebral body", "disc", "spinal canal"],
        "resolution_mm": args.res,
        "canvas_px": CANVAS,
        "train_patch_px": PATCH,
        "para_mm": args.para_mm,
        "training_volumes": n_vol,
        "training_slices": len(x),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": seed,
        "device": str(device),
        "loss_per_epoch": history,
        "seconds": round(time.time() - t0, 1),
        "git_commit": git_commit(),
        "versions": versions(),
        "command": " ".join(sys.argv),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    log.info(
        "exported %s (%.2f MB)", out / "labeller_v1.onnx", (out / "labeller_v1.onnx").stat().st_size / 1e6
    )


def export_onnx(model: Any, path: Path) -> None:
    torch = _torch()
    dummy = torch.zeros(1, 1, PATCH, PATCH)
    kwargs: dict[str, Any] = {}
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        kwargs["dynamo"] = False
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
        **kwargs,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Anatomy metrics (spinescope.anatomy._validate) and slice segmentation Dice on a SPIDER subset."""
    from spinescope import load_study
    from spinescope.anatomy._validate import aggregate, score_anchors, truth_from_mask
    from spinescope.anatomy.ml import label_spine_ml, load_session, segment_stack
    from spinescope.io.volumes import read_volume
    from spinescope.series import to_display_orientation

    root = spider_root(args.spider)
    rows = rows_for(root, args.subset)
    if args.limit:
        rows = rows[: args.limit]
    session = load_session(args.model)
    per_vol: list[dict[str, Any]] = []
    scores = []
    dice_acc = {c: [0.0, 0.0] for c in range(1, N_CLASSES)}  # per class: 2|P&G|, |P|+|G|
    t0 = time.time()
    for i, r in enumerate(rows):
        name = r["new_file_name"]
        img_path = str(root / "images" / f"{name}.mha")
        mask_path = str(root / "masks" / f"{name}.mha")
        truth, mx = truth_from_mask(mask_path)
        study = load_study(img_path)
        anchors = label_spine_ml(study, session=session)
        sc = score_anchors(anchors, truth, mx)
        scores.append(sc)
        seq = "T1" if name.endswith("t1") else ("T2 SPACE" if "SPACE" in name else "T2")
        per_vol.append(
            {
                "name": name,
                "sequence": seq,
                "field_T": r.get("MagneticFieldStrength", ""),
                "vendor": r.get("Manufacturer", ""),
                "matched": sc.matched,
                "n_truth": sc.n_truth,
                "n_pred": sc.n_pred,
                "label_correct": sc.label_correct,
                "centre_err_median_mm": float(np.median(sc.centre_err_mm)) if sc.centre_err_mm else None,
            }
        )
        # Slice segmentation Dice on the fixed grid (all slices of the chosen sagittal stack).
        stack = study.best("sag")
        if stack is not None:
            seg = segment_stack(stack, session=session)
            mstack = to_display_orientation(read_volume(mask_path))
            gt_cls = class_map(mstack.vol)
            pred = seg.probs.argmax(axis=1)
            for k, g in enumerate(seg.grids):
                gt = np.rint(resample_slice(mstack, g, array=gt_cls, order=0)).astype(np.int64)
                for c in range(1, N_CLASSES):
                    p_c = pred[k] == c
                    g_c = gt == c
                    dice_acc[c][0] += 2.0 * float(np.logical_and(p_c, g_c).sum())
                    dice_acc[c][1] += float(p_c.sum() + g_c.sum())
        if (i + 1) % 10 == 0:
            log.info("evaluated %d/%d (%.0f s)", i + 1, len(rows), time.time() - t0)
    res: dict[str, Any] = {
        "subset": args.subset,
        "volumes": len(rows),
        "overall": aggregate(scores),
        "dice": {CLASS_NAMES[c]: (a / b if b else float("nan")) for c, (a, b) in dice_acc.items()},
        "per_volume": per_vol,
    }
    groups: dict[str, dict[str, Any]] = {}
    for key in ("sequence", "field_T", "vendor"):
        groups[key] = {}
        for val in sorted({str(v[key]) for v in per_vol}):
            idx = [j for j, v in enumerate(per_vol) if str(v[key]) == val]
            groups[key][val or "unknown"] = aggregate([scores[j] for j in idx])
    res["by_group"] = groups
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"eval_{args.subset}.json").write_text(json.dumps(res, indent=1))
    log.info("wrote %s", Path(args.out) / f"eval_{args.subset}.json")
    print(json.dumps({k: res[k] for k in ("subset", "volumes", "overall", "dice")}, indent=1))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spider", default=None, help="SPIDER root (default $SPINESCOPE_DATA/spider)")
    ap.add_argument("--out", default="/tmp/spinescope-anatomy_ml", help="scratch output directory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train on the training subset and export ONNX")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=12)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=20260)
    t.add_argument(
        "--para-mm", type=float, default=15.0, help="slices within this distance of the canal centre"
    )
    t.add_argument("--res", type=float, default=DEFAULT.ml_resolution_mm)
    t.add_argument("--limit", type=int, default=0, help="debug: first N volumes only")
    t.add_argument("--rebuild", action="store_true", help="ignore the slice cache")
    t.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    e = sub.add_parser("evaluate", help="metrics on a subset (default validation)")
    e.add_argument("--subset", default="validation")
    e.add_argument("--model", default=None, help="ONNX file (default: packaged weights)")
    e.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
