"""Temporal U-Net and transformer association, adapted from pilkwang's Biohub support pack.

Harmonic association follows flexonafft's Biohub Harmonic Fusion, using the
mutual-support rule from yusuketogashi's Biohub notebook (v18, CC0).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

import tracksdata as td
from .v1284_coordinate_refinement import refine as _v1284_refine

from biohub_tracking.io import open_dataset

from .tracking_model import (
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)
from biohub_tracking.models import TemporalUNet3D

INTERACTIVE = False


# =============================================================================
# Prediction config
# =============================================================================

@dataclass
class PredictConfig:
    """Detection, association, and ILP parameters.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.5
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in µm for detection peak extraction
    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # ILP imposes parent and child limits jointly across frames.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2


# =============================================================================
# Helpers
# =============================================================================


def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> td.graph.InMemoryGraph:
    """Build a graph with node centroids and association probabilities."""
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])

    return graph


# =============================================================================
# Model loading
# =============================================================================

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample


# =============================================================================
# Per-frame loading
# =============================================================================

def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum sigmoid probability for a peak to be considered (default 0.5).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


def _average_xy_views(model, images, features, logits):
    """Average the original and seven aligned XY transforms in submission order."""
    feature_sum = features.clone()
    count = 1
    for dims in [(-1,), (-2,), (-2, -1)]:
        view = images.flip(dims)
        transformed_features, transformed_logits = model.encode(view)
        for frame in range(len(logits)):
            logits[frame] = logits[frame] + transformed_logits[frame].flip(dims)
        feature_sum = feature_sum + transformed_features.flip(dims)
        del view, transformed_logits, transformed_features
        count += 1
    for turns in (1, 3):
        view = torch.rot90(images, turns, dims=(-2, -1))
        transformed_features, transformed_logits = model.encode(view)
        for frame in range(len(logits)):
            logits[frame] = logits[frame] + torch.rot90(transformed_logits[frame], -turns, dims=(-2, -1))
        feature_sum = feature_sum + torch.rot90(transformed_features, -turns, dims=(-2, -1))
        del view, transformed_logits, transformed_features
        count += 1
    view = images.transpose(-1, -2)
    transformed_features, transformed_logits = model.encode(view)
    for frame in range(len(logits)):
        logits[frame] = logits[frame] + transformed_logits[frame].transpose(-1, -2)
    feature_sum = feature_sum + transformed_features.transpose(-1, -2)
    del view, transformed_logits, transformed_features
    count += 1
    view = torch.rot90(images, 1, dims=(-2, -1)).transpose(-1, -2)
    transformed_features, transformed_logits = model.encode(view)
    for frame in range(len(logits)):
        aligned = torch.rot90(transformed_logits[frame].transpose(-1, -2), -1, dims=(-2, -1))
        logits[frame] = logits[frame] + aligned
    feature_sum = feature_sum + torch.rot90(transformed_features.transpose(-1, -2), -1, dims=(-2, -1))
    del view, transformed_logits, transformed_features
    count += 1
    return feature_sum / count, [frame_logits / count for frame_logits in logits]


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    downsample: tuple[int, ...] = (1, 4, 4),
    secondary_model: UNetNodeTransformer | None = None,
    secondary_edge_weight: float = 0.0,
    secondary_detection_weight: float = 0.0,
    secondary_low_margin_max: float = 0.2,
    low_detection_threshold: float = 0.0,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], np.ndarray, np.ndarray]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once.  UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    low_coords, low_scores : optional below-threshold peaks and their probabilities
    """
    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)  # for coord rescaling at the end
    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []
    low_detections: list[tuple[np.ndarray, np.ndarray]] = []

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 → T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in tqdm(
        window_starts,
        desc="  windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1) ---
        imgs = torch.stack([
            _load_frame(zarr_arr, t, target_shape, downsample)
            for t in frame_indices
        ])  # (W, *spatial)
        # Quantile normalisation (0.1%–99.9%) to match training pipeline.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)   # (1, W, *spatial)

        unet_out, det_logits = model.encode(imgs)
        if cfg.det_tta:
            unet_out, det_logits = _average_xy_views(model, imgs, unet_out, det_logits)

        secondary_unet_out = None
        if secondary_model is not None:
            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)
            if secondary_detection_weight > 0.0:
                if cfg.det_tta:
                    mean_features, secondary_det_logits = _average_xy_views(
                        secondary_model, imgs, secondary_unet_out, secondary_det_logits,
                    )
                    secondary_unet_out = 0.25 * secondary_unet_out + 0.75 * mean_features
                    del mean_features

                for f in range(W):
                    primary_det = det_logits[f]
                    secondary_det = secondary_det_logits[f]
                    primary_mean = primary_det.mean()
                    secondary_mean = secondary_det.mean()
                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)
                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)
                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                    secondary_det_aligned = (
                        (secondary_det - secondary_mean) * scale_ratio + primary_mean
                    )
                    blended_det = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )
                    primary_candidates = len(_detect_cells_pooled(
                        primary_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    blended_candidates = len(_detect_cells_pooled(
                        blended_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    minimum_retention = 0.90
                    candidate_retention = (
                        blended_candidates / primary_candidates
                        if primary_candidates
                        else 1.0
                    )
                    use_primary_detection = bool(
                        primary_candidates > 0
                        and candidate_retention < minimum_retention
                    )
                    det_logits[f] = (
                        primary_det if use_primary_detection else blended_det
                    )


            del secondary_det_logits

        del imgs

        # --- Detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(
                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,
                )
                if low_detection_threshold > 0:
                    low = _detect_cells_pooled(det_logits[f_idx][0], t, low_detection_threshold, pool_k)
                    if len(low):
                        logits = det_logits[f_idx][0][0]
                        indices = torch.as_tensor(low[:, 1:].astype(np.int64), device=logits.device)
                        scores = torch.sigmoid(logits[indices[:, 0], indices[:, 1], indices[:, 2]])
                        low_detections.append((low.astype(np.float32), scores.float().cpu().numpy()))
                arr = _v1284_refine(ds_path, t, arr, unet_out[:, f_idx])
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Build tensors (batch_size=1).
            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.
            window_shape = (W,) + image_shape[1:]
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            unet_feat_src = model._index_features(
                unet_out[:, f_idx], p_coords_src, p_mask_src,
            )
            unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
            )
            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)

            _bidirectional_weight = 0.15
            if _bidirectional_weight > 0.0:
                reverse_logits_native = model.predict_edges(
                    unet_feat_tgt, unet_feat_src,
                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,
                    p_pos_tgt, p_pos_src,
                    p_mask_tgt, p_mask_src,
                )  # (1, n_tgt, n_src)
                reverse_logits_pair = reverse_logits_native.transpose(1, 2)

                forward_center = edge_logits_pair.mean(dim=1, keepdim=True)
                forward_scale = edge_logits_pair.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)
                reverse_scale = reverse_logits_pair.float().std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)
                reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)
                reverse_aligned = (
                    (reverse_logits_pair - reverse_center) * reverse_scale_ratio
                    + forward_center
                )
                # Require mutual forward/reverse support in probability space.
                # The harmonic mean penalizes a candidate when either temporal direction
                # assigns it very low probability, while calibration preserves the forward
                # logit scale used by the unchanged downstream candidate threshold and ILP.
                forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)
                reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
                harmonic_prob = 1.0 / (
                    (1.0 - _bidirectional_weight) / forward_prob
                    + _bidirectional_weight / reverse_prob
                )
                harmonic_prob = harmonic_prob / harmonic_prob.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
                harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
                harmonic_scale = harmonic_logits.std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-4)
                harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
                edge_logits_pair = (
                    (harmonic_logits - harmonic_center) * harmonic_scale_ratio
                    + forward_center
                ).to(reverse_aligned.dtype)
                del (
                    reverse_logits_native,
                    reverse_logits_pair,
                    reverse_aligned,
                    forward_prob,
                    reverse_prob,
                    harmonic_prob,
                    harmonic_logits,
                )
            if secondary_model is not None:
                if secondary_unet_out is None:
                    raise RuntimeError("Secondary model is loaded but its feature map is missing")
                secondary_feat_src = secondary_model._index_features(
                    secondary_unet_out[:, f_idx], p_coords_src, p_mask_src,
                )
                secondary_feat_tgt = secondary_model._index_features(
                    secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
                )
                secondary_logits_pair = secondary_model.predict_edges(
                    secondary_feat_src, secondary_feat_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt,
                    p_mask_src, p_mask_tgt,
                )

                primary_center = edge_logits_pair.mean(dim=1, keepdim=True)
                primary_scale = edge_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)
                secondary_scale = secondary_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                secondary_for_mix = (secondary_logits_pair - secondary_center) * scale_ratio + primary_center
                blend_weight = 0.0
                if n_src >= 2:
                    primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
                    secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
                    primary_top2 = torch.topk(primary_probs, k=2, dim=0)
                    secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
                    primary_margin = primary_top2.values[0] - primary_top2.values[1]
                    same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                    uncertainty = ((secondary_low_margin_max - primary_margin) / secondary_low_margin_max).clamp(0.0, 1.0)
                    local_weight = secondary_edge_weight * uncertainty
                    local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
                    blend_weight = local_weight.view(1, 1, -1)

                edge_logits_pair = (
                    (1.0 - blend_weight) * edge_logits_pair
                    + blend_weight * secondary_for_mix
                )

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

        del unet_out
        if secondary_unet_out is not None:
            del secondary_unet_out

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    if low_detections:
        low_coords = np.concatenate([item[0] for item in low_detections]).astype(np.float32)
        low_coords[:, 1:] *= ds_arr
        low_coords = low_coords.astype(np.int16)
        low_scores = np.concatenate([item[1] for item in low_detections]).astype(np.float32)
    else:
        low_coords = np.empty((0, 4), np.int16)
        low_scores = np.empty(0, np.float32)
    return coords, all_edges, low_coords, low_scores
