"""Subvoxel coordinate regression on frozen U-Net features.

The head architecture, feature normalization, bounded offset and trilinear
indexing follow anvithpothula's Biohub x138; the head ensemble was trained for
this solution. Coordinates enter on the 1.625 µm detection grid.
"""
import os
from pathlib import Path

import numpy as np
import torch

SPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)      # µm на шаг downsample-сетки
GT_VOXEL = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)  # µm на воксель исходной сетки
OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
MAX_SHIFT_UM = 2.0
_CACHE = None
_GT_CACHE = {}


def make_head():
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x, max_shift: float = MAX_SHIFT_UM):
    """Map the predicted offset to a vector shorter than max_shift µm."""
    delta = head(x)
    return max_shift * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def sample_features(feature, arr):
    xyz = torch.as_tensor(np.asarray(arr[:, 1:]), device=feature.device, dtype=torch.long)
    blocks = []
    for offset in OFFSETS:
        loc = xyz + torch.tensor(offset, device=feature.device)
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size - 1)
        blocks.append(feature[0, :, loc[:, 0], loc[:, 1], loc[:, 2]].T)
    # Центральное представление + направленные разности.
    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)


def index_features(self, maps, coords, mask):
    """Трилинейная выборка; на целых координатах ПОБИТОВО воспроизводит родной gather."""
    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)
    for batch in range(len(maps)):
        n = int(mask[batch].sum())
        if not n:
            continue
        q = coords[batch, :n].clone()
        for axis, size in enumerate(maps.shape[-3:]):
            q[:, axis].clamp_(0, size - 1)
        low = q.floor().long()
        frac = q - low
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    shift = torch.tensor([z, y, x], device=maps.device)
                    loc = low + shift
                    for axis, size in enumerate(maps.shape[-3:]):
                        loc[:, axis].clamp_(0, size - 1)
                    weight = torch.where(shift.bool(), frac, 1 - frac).prod(dim=1)
                    out[batch, :n] += maps[batch, :, loc[:, 0], loc[:, 1], loc[:, 2]].T * weight[:, None]
    return out


# --------------------------------------------------------------------------- #
#  capture: сопоставление с GT и запись только сопоставленных пар
# --------------------------------------------------------------------------- #
def _gt_frames(stem: str):
    """{t: (M, 3) координаты GT-узлов кадра в µm} для клипа `stem` (или None)."""
    if stem in _GT_CACHE:
        return _GT_CACHE[stem]
    out = None
    root = os.environ.get("V1284_GT", "")
    try:
        if root:
            import zarr
            g = zarr.open(str(Path(root) / f"{stem}.geff"), mode="r")
            t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
            zyx = np.stack([np.asarray(g[f"nodes/props/{a}/values"][:]).astype(np.float32)
                            for a in "zyx"], axis=1) * GT_VOXEL
            out = {int(tt): zyx[t == tt] for tt in np.unique(t)}
            print(f"V1284 capture: GT {stem}: {len(t)} узлов, {len(out)} кадров", flush=True)
    except Exception as exc:                                    # noqa: BLE001 — capture не должен ронять прогон
        print(f"V1284 capture: GT {stem} недоступен ({type(exc).__name__}: {exc})", flush=True)
        out = None
    _GT_CACHE[stem] = out
    return out


def _capture(stem: str, t: int, arr, x):
    folder = Path(os.environ["V1284_CAPTURE"]) / stem
    folder.mkdir(parents=True, exist_ok=True)
    gt = _gt_frames(stem)
    feats = x.cpu().numpy().astype(np.float16)
    if gt is None:                                              # без GT — сырой дамп кадра
        np.savez_compressed(folder / f"{int(t):04d}.npz", coords=arr, features=feats)
        return
    g = gt.get(int(t))
    if g is None or not len(g):
        return
    from scipy.spatial import cKDTree
    d_um = np.asarray(arr[:, 1:], dtype=np.float32) * SPACING
    match_um = float(os.environ.get("V1284_MATCH_UM", "4.0"))
    di, gi = cKDTree(g).query(d_um)                             # ближайший GT для каждой детекции
    dj, dj_i = cKDTree(d_um).query(g)                           # ближайшая детекция для каждого GT
    keep = np.nonzero((di <= match_um) & (dj_i[gi] == np.arange(len(d_um))))[0]   # взаимно ближайшие
    if not len(keep):
        return
    cap = int(os.environ.get("V1284_CAP_PER_FRAME", "48"))
    if cap and len(keep) > cap:
        rng = np.random.default_rng(abs(hash((stem, int(t)))) % (2 ** 32))
        keep = np.sort(rng.choice(keep, cap, replace=False))
    delta = (g[gi[keep]] - d_um[keep]).astype(np.float32)
    np.savez_compressed(folder / f"{int(t):04d}.npz", features=feats[keep],
                        delta=delta, dist=di[keep].astype(np.float32),
                        coords=np.asarray(arr[keep], dtype=np.int16))


def refine(ds_path, t, arr, feature):
    global _CACHE
    mode = os.environ["V1284_MODE"]
    if not len(arr):
        return arr
    if mode == "zero":
        return arr.astype(np.float32)
    x = sample_features(feature, arr).float()
    if mode == "capture":
        _capture(ds_path.stem, int(t), arr, x)
        return arr
    if _CACHE is None:
        saved = torch.load(os.environ["V1284_HEAD"], map_location="cpu", weights_only=True)
        _sds = saved.get("state_dicts") or [saved["state_dict"]]
        _heads = []
        for _sd in _sds:
            _h = make_head().to(feature.device)
            _h.load_state_dict(_sd)
            _h.eval()
            _heads.append(_h)
        _CACHE = (_heads, saved["mean"].to(feature.device), saved["scale"].to(feature.device),
                  float(saved.get("max_shift", MAX_SHIFT_UM)))
    heads, mean, scale, max_shift = _CACHE
    with torch.no_grad():
        _xn = (x - mean) / scale
        _acc = bounded(heads[0], _xn, max_shift)
        for _h in heads[1:]:
            _acc = _acc + bounded(_h, _xn, max_shift)
        shift = (_acc / len(heads)).cpu().numpy() / SPACING
    result = arr.astype(np.float32).copy()
    result[:, 1:] += shift
    result[:, 1:] = np.clip(result[:, 1:], 0, np.asarray(feature.shape[-3:]) - 1)
    if (not np.isfinite(result).all()
            or np.max(np.linalg.norm((result[:, 1:] - arr[:, 1:]) * SPACING, axis=1)) > max_shift + 1e-5):
        raise RuntimeError("invalid V1284 displacement")
    return result
