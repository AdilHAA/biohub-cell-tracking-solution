"""Run the Biohub cell tracking solution and write a submission CSV."""

import argparse
import contextlib
import csv
import os
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time


ROOT = Path(__file__).resolve().parent
COLUMNS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("v92", "v96"), default="v92")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the Zarr movies")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--deepcenter", type=Path, required=True)
    parser.add_argument("--coordinate-head", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    mode = os.environ.get("V1284_MODE", "candidate")
    if mode not in {"candidate", "capture", "zero"}:
        parser.error(f"Unknown coordinate refinement mode: {mode}")
    if mode == "candidate" and args.coordinate_head is None:
        parser.error("--coordinate-head is required for coordinate refinement")
    if mode == "capture" and not os.environ.get("V1284_CAPTURE"):
        parser.error("Capture mode requires V1284_CAPTURE")
    return args, mode


def _read_graph(graph):
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id, "t": int(row["t"]),
            **{axis: float(row[axis]) for axis in "zyx"},
        }
    edges = [
        {"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
         "edge_prob": None if row.get("edge_prob") is None else float(row["edge_prob"])}
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return nodes, edges


def _write_graph(writer, dataset, nodes, edges, row_id):
    if not nodes:
        raise ValueError(f"{dataset}: no nodes in the repaired graph")
    for node_id in sorted(nodes):
        node = nodes[node_id]
        writer.writerow((row_id, dataset, "node", node_id, node["t"],
                         *(max(0, int(round(node[axis]))) for axis in "zyx"), -1, -1))
        row_id += 1
    for edge in edges:
        src, tgt = edge["source_id"], edge["target_id"]
        if src not in nodes or tgt not in nodes:
            raise ValueError(f"{dataset}: dangling edge {src} -> {tgt}")
        writer.writerow((row_id, dataset, "edge", -1, -1, -1, -1, -1, src, tgt))
        row_id += 1
    return row_id


def _predict(args, movies, folder):
    import numpy as np
    import torch
    import tracksdata as td

    from upstream.scripts.predict_tracks import PredictConfig, build_graph, load_model, predict_video

    device = torch.device(args.device)
    primary, window, downsample = load_model(args.primary, device)
    secondary, secondary_window, secondary_downsample = load_model(args.secondary, device)
    if (window, downsample) != (secondary_window, secondary_downsample):
        raise ValueError("The two temporal models use different inference grids")
    config = PredictConfig(det_threshold=0.965, threshold=0.48, use_ilp=True,
                           ilp_edge_weight=-1.0, ilp_appearance_weight=0.0,
                           ilp_disappearance_weight=2.0, ilp_division_weight=1.2)
    for movie in movies:
        coords, edges, low_coords, low_scores = predict_video(
            primary, movie, device, config, window_size=window, downsample=downsample,
            secondary_model=secondary, secondary_edge_weight=0.15, secondary_detection_weight=0.80,
            secondary_low_margin_max=0.35, low_detection_threshold=0.30 if args.variant == "v92" else 0.0,
        )
        graph = build_graph(coords, edges)
        if graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=config.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=config.ilp_appearance_weight,
                disappearance_weight=config.ilp_disappearance_weight,
                division_weight=config.ilp_division_weight,
                **({"timeout": 1200.0} if args.variant == "v92" else {}),
            )
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                graph = solver.solve(graph)
        graph.to_geff(folder / f"{movie.stem}.geff")
        if args.variant == "v92":
            np.savez(folder / f"{movie.stem}.npz", low_coords=low_coords, low_scores=low_scores)
        print(f"{movie.stem}: {graph.num_nodes()} nodes, {graph.num_edges()} edges after ILP", flush=True)


def _repair(args, movies, folder, started):
    import numpy as np
    import tracksdata as td

    from postprocess import GraphRepair, load_deepcenter

    repair = GraphRepair(args.data_dir, args.variant)
    deepcenter = load_deepcenter(args.deepcenter, repair.settings.DEEPCENTER_EXPECTED_EPOCH, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission = folder / "submission.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        row_id = 0
        for movie in movies:
            dataset = movie.stem
            graph = td.graph.IndexedRXGraph.from_geff(folder / f"{dataset}.geff")
            nodes, edges = _read_graph(graph[0] if isinstance(graph, tuple) else graph)
            if args.variant == "v92":
                with np.load(folder / f"{dataset}.npz", allow_pickle=False) as peaks:
                    repair.low_detections = (peaks["low_coords"], peaks["low_scores"])
                if time.monotonic() - started > 27_000:
                    repair.disable_expensive_stages()
                raw_nodes = {node_id: node.copy() for node_id, node in nodes.items()}
                raw_edges = [edge.copy() for edge in edges]
            try:
                nodes, edges, _ = repair.filter_output_graph(nodes, edges, dataset=dataset, deepcenter_bundle=deepcenter)
                if not nodes:
                    raise ValueError("Graph repair removed every node")
            except Exception as exc:
                if args.variant != "v92":
                    raise
                print(f"{dataset}: {exc}; keeping the filtered ILP graph", file=sys.stderr, flush=True)
                nodes, edges, _ = repair.fallback_output_graph(raw_nodes, raw_edges)
            row_id = _write_graph(writer, dataset, nodes, edges, row_id)
    # Copy only after all movies are complete; temporary GEFFs stay outside the output tree.
    shutil.copyfile(submission, args.output)
    print(f"Wrote {row_id:,} rows to {args.output}")


def main():
    started = time.monotonic()
    args, mode = parse_args()
    for path in (args.primary, args.secondary, args.deepcenter, args.coordinate_head):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    movies = sorted(args.data_dir.glob("*.zarr"))
    if not movies:
        raise FileNotFoundError(f"No Zarr movies in {args.data_dir}")
    sys.path.insert(0, str(ROOT / "upstream" / "src"))
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    os.environ["V1284_MODE"] = mode
    if args.coordinate_head is not None:
        os.environ["V1284_HEAD"] = str(args.coordinate_head.resolve())
    with TemporaryDirectory(prefix="biohub-") as temp:
        folder = Path(temp)
        _predict(args, movies, folder)
        _repair(args, movies, folder, started)


if __name__ == "__main__":
    main()
