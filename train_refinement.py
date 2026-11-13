"""Train the coordinate refinement ensemble used by v92 and v96."""

import argparse
from pathlib import Path

import numpy as np
import torch

from upstream.scripts.v1284_coordinate_refinement import bounded, make_head

EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_SHIFT_UM = 2.0
ENSEMBLE = (
    (2.385, (2718, 1337, 7, 11, 22, 33, 101, 4242)),
    (2.0, (2718, 1337, 7, 101, 4242)),
)


def load_captures(inputs):
    """Read merged arrays or per-frame captures in clip and frame order."""
    if len(inputs) == 1 and inputs[0].is_file():
        with np.load(inputs[0], allow_pickle=False) as data:
            return data['base'].astype(np.float32), data['delta'].astype(np.float32), data['clip'].astype(str)

    clips = sorted((clip.name, clip) for root in inputs for clip in root.iterdir() if clip.is_dir())
    if len({name for name, _ in clips}) != len(clips):
        raise ValueError('Each clip must appear in only one capture directory')
    features, targets, names = [], [], []
    for name, clip in clips:
        for frame in sorted(clip.glob('*.npz')):
            with np.load(frame, allow_pickle=False) as data:
                if 'delta' not in data:
                    continue
                key = 'base224' if 'base224' in data else 'features'
                features.append(data[key].astype(np.float32))
                targets.append(data['delta'].astype(np.float32))
                names.extend([name] * len(data['delta']))
    if not targets:
        raise ValueError('No matched detection and ground truth pairs found')
    return np.concatenate(features), np.concatenate(targets), np.asarray(names)


def fit_head(features, targets, weights, seed, device, epochs=EPOCHS):
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    head = make_head().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    head.train()
    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator).to(device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            residual = bounded(head, features[batch], MAX_SHIFT_UM) - targets[batch]
            loss = (weights[batch] * residual.square().sum(dim=-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
    return head.eval()


def train_ensemble(features, targets, clips, device, epochs=EPOCHS):
    if features.shape != (len(clips), 224) or targets.shape != (len(clips), 3) or not len(clips):
        raise ValueError('Expected nonempty arrays of 224 features and 3 offsets per detection')
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-3] = 1.0
    x = torch.as_tensor((features - mean) / scale, device=device)
    y = torch.as_tensor(targets, device=device)
    embryo = np.char.startswith(clips.astype(str), '44b6')
    states, seeds, balances = [], [], []
    for balance, group in ENSEMBLE:
        weights = np.where(embryo, balance, 1.0).astype(np.float32)
        w = torch.as_tensor(weights / weights.mean(), device=device)
        for seed in group:
            head = fit_head(x, y, w, seed, device, epochs=epochs)
            states.append({key: value.cpu() for key, value in head.state_dict().items()})
            seeds.append(seed)
            balances.append(balance)
    return {
        'state_dicts': states,
        'mean': torch.as_tensor(mean),
        'scale': torch.as_tensor(scale),
        'max_shift': MAX_SHIFT_UM,
        'seeds': seeds,
        'w44_per_net': balances,
        'ensemble': 'mean_of_bounded_shifts',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    features, targets, clips = load_captures(args.captures)
    ensemble = train_ensemble(features, targets, clips, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ensemble, args.output)


if __name__ == '__main__':
    main()
