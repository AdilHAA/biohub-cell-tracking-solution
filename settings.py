"""Fixed post-processing settings for the v92 and v96 submissions."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RepairSettings:
    OUTPUT_EDGE_MAX_UM: float = 14.0
    OUTPUT_ENFORCE_NEXT_FRAME: bool = True
    OUTPUT_SINGLE_PARENT_REPAIR: bool = True
    OUTPUT_SINGLE_CHILD_REPAIR: bool = False
    OUTPUT_PRUNE_ISOLATED: bool = True
    OUTPUT_MOTION_RELINK: bool = True
    MOTION_RELINK_TIGHT_UM: float = 6.0
    MOTION_RELINK_RELAXED_UM: float = 10.0
    MOTION_RELINK_VELOCITY_WEIGHT: float = 0.25
    MOTION_RELINK_FLOW_MODE: str = 'seed'
    MOTION_RELINK_FLOW_K: int = 12
    MOTION_RELINK_FLOW_RADIUS_UM: float = 40.0
    MOTION_RELINK_FLOW_EXCLUDE_UM: float = 1.5
    MOTION_RELINK_FLOW_MIN_SAMPLES: int = 4
    MOTION_RELINK_FLOW_GATE: bool = True
    MOTION_RELINK_FLOW_ITER: int = 1
    MOTION_RELINK_FLOW_SEED_GATE_UM: float = 0.0
    MOTION_RELINK_FLOW_RAW_ADMIT: bool = True
    MOTION_RELINK_FLOW_Z_WEIGHT: float = 1.0
    MOTION_RELINK_FLOW_RAW_COST: float = 0.0
    MOTION_RELINK_FLOW_TIGHT_UM: float = 7.0
    MOTION_RELINK_FLOW_RELAXED_UM: float = 0.0
    READMIT_RADIUS_UM: float = 4.0
    READMIT_MIN_SCORE: float = 0.965
    GAPFILL_MAX_GAP: int = 3
    GAPFILL_MIN_SCORE: float = 0.5
    GAPFILL_STEP_UM: float = 5.0
    GAPFILL_PEAK_RADIUS_UM: float = 3.5
    GAPFILL_EXCLUDE_UM: float = 2.0
    GAPFILL_ALLOW_SYNTHETIC: int = 0
    GAPFILL_CONTEXT: bool = True
    GAPFILL_MAX_ADDED_FRAC: float = 0.03
    MOTION_RELINK_LEARNED_BONUS: float = 1.0
    MOTION_RELINK_MAX_FRAME_NODES: int = 2600
    OUTPUT_DIVISION_GEOMETRY_FILTER: bool = False
    DIV_PARENT_MAX_UM: float = 10.5
    DIV_SISTER_MAX_UM: float = 8.0
    DIV_DROP_TO_SINGLE_IF_BAD: bool = True
    OUTPUT_GAP_CLOSE: bool = True
    GAP_CLOSE_MAX_GAP: int = 2
    GAP_CLOSE_UM: float = 5.0
    GAP_DENSITY_ADAPTIVE: bool = True
    GAP_DENSITY_REFERENCE_UM: float = 6.5
    GAP_DENSITY_GAIN: float = 0.04
    GAP_DENSITY_MAX_STEP_DELTA_UM: float = 0.125
    GAP_DENSITY_NEIGHBORS: int = 3
    GAP_CLOSE_REUSE_EXISTING: bool = True
    GAP_CLOSE_REUSE_UM: float = 3.2
    GAP_CLOSE_MAX_ADDED_FRAC: float = 0.05
    GAP_CLOSE_MAX_ADDED_ABS: int = 2000
    GAP_REFINE_SYNTHETIC: bool = True
    GAP_REFINE_WIN_Z: int = 1
    GAP_REFINE_WIN_YX: int = 3
    GAP_REFINE_MAX_SHIFT_UM: float = 3.2
    OUTPUT_FILTER_SHORT_TRACKS: bool = True
    OUTPUT_MIN_TRACK_LEN: int = 6
    OUTPUT_KEEP_DIVISION_COMPONENTS: bool = True
    ADAPTIVE_SHORT_TRACK_RESCUE: bool = True
    SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC: float = 0.1
    SHORT_TRACK_RESCUE_MIN_LEN: int = 4
    SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB: float = 0.88
    SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM: float = 3.0
    SHORT_TRACK_RESCUE_MAX_NODES_FRAC: float = 0.012
    SHORT_TRACK_RESCUE_MAX_NODES_ABS: int = 120
    OUTPUT_LINEFIT_SMOOTH: bool = True
    OUTPUT_LINEFIT_WEIGHT: float = 0.8
    OUTPUT_LINEFIT_WINDOW: int = 2
    OUTPUT_GAP2_RECOVERY: bool = True
    GAP2_MAX_TOTAL_UM: float = 10.2
    GAP2_MAX_STEP_UM: float = 4.4
    GAP2_MAX_LINKS_FRAC: float = 0.0045
    GAP2_MAX_LINKS_ABS: int = 180
    GAP2_REQUIRE_CONTEXT: bool = True
    GAP2_FRAME_FRAC_CAP: float = 0.006
    OUTPUT_SAFE_DIVISIONS: bool = True
    SAFE_DIV_MAX_UM: float = 9.0
    SAFE_DIV_SISTER_MAX_UM: float = 14.0
    SAFE_DIV_SISTER_SYMMETRY_TAU: float = 0.6
    SAFE_DIV_EXISTING_CHILD_MAX_UM: float = 10.0
    SAFE_DIV_FRAME_FRAC_CAP: float = 0.0076
    SAFE_DIV_GLOBAL_FRAC_CAP: float = 0.00375
    SAFE_DIV_DIVERGE_UM: float = 2.25
    SAFE_DIV_REQUIRE_DIVERGENCE: bool = True
    SAFE_DIV_REQUIRE_MUTUAL_NN: bool = True
    USE_DEEPCENTER_VETO: bool = True
    DEEPCENTER_GAP_VETO: bool = True
    DEEPCENTER_SAFE_DIV_VETO: bool = True
    DEEPCENTER_GAP_THRESHOLD: float = 0.25
    DEEPCENTER_EXPECTED_EPOCH: int = 2
    DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM: float = 8.5
    DEEPCENTER_SAFE_DIV_THRESHOLD: float = 0.25
    DEEPCENTER_SCORE_WIN_Z: int = 1
    DEEPCENTER_SCORE_WIN_YX: int = 2
    DEEPCENTER_SCORE_CACHE_MAX_FRAMES: int = 8
    DEEPCENTER_TTA: bool = True
    FRAME_CACHE_MAX_FRAMES: int | None = None


def repair_settings(variant):
    settings = RepairSettings()
    if variant == "v92":
        return replace(
            settings,
            MOTION_RELINK_TIGHT_UM=5.5,
            MOTION_RELINK_VELOCITY_WEIGHT=0.5,
            FRAME_CACHE_MAX_FRAMES=48,
        )
    if variant == "v96":
        return settings
    raise ValueError(f"Unknown submission variant: {variant}")
