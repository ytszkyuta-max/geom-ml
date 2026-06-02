"""
曲率特徴の「真の寄与」を切り分ける3群（+対照）ablation

前回(exp_holonomy_ablation.py)で曲率特徴がMSE49%改善したが、合成データでは
曲率が「メッシュ上の位置ラベル」として効いている疑いがあった。
そこで入力に足す2列の補助特徴の「中身」だけを変えて公平に比較する:

  - baseline : 補助なし（気象変数のみ）
  - position : 位置特徴2列 [頂点次数(正規化), 緯度(正規化)]  ← 位置ヒントの対照群
  - random   : ランダム2列（頂点ごと固定）  ← 「ただの2列増」の対照群
  - holonomy : 曲率特徴2列 [cosΘ̄, sinΘ̄]

position と holonomy の差が「曲率特有の寄与」。
random との差が「情報を持つ特徴 vs ノイズ」。

使い方:
  python exp_holonomy_3way.py --level 1 --seeds 0 1 2 3 4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from icosahedron import (build_icosahedron, subdivide, build_gauge_data_general,
                         compute_local_frames, compute_connection_angles,
                         compute_plaquette_holonomy, scatter_holonomy_to_vertices)
from gauge_cnn import IcosGaugeCNNGeneral
from era5_loader import make_synthetic_era5, xyz_to_latlon
from era5_task import MeshForecastDataset, train_one_epoch, evaluate, _persistence_loss


def holonomy_features(verts, faces):
    edge_set = set()
    for f in faces:
        for i in range(3):
            edge_set.add(tuple(sorted([int(f[i]), int(f[(i + 1) % 3])])))
    edges = np.array(sorted(edge_set), dtype=np.int64)
    frames = compute_local_frames(verts)
    angles = compute_connection_angles(verts, edges, frames)
    holo = compute_plaquette_holonomy(faces, edges, angles)
    return scatter_holonomy_to_vertices(faces, holo, len(verts)).astype(np.float32)  # (V,2)


def position_features(verts, nb_mask):
    """位置特徴 [頂点次数(標準化), 緯度(標準化)] (V,2)。"""
    deg = nb_mask.sum(axis=1)                       # (V,)
    lats, _ = xyz_to_latlon(verts)                  # (V,)
    def z(a):
        a = a.astype(np.float32)
        return (a - a.mean()) / (a.std() + 1e-6)
    return np.stack([z(deg), z(lats)], axis=1).astype(np.float32)  # (V,2)


def make_aux(kind, verts, faces, nb_mask, V):
    if kind == 'baseline':
        return None
    if kind == 'holonomy':
        return holonomy_features(verts, faces)
    if kind == 'position':
        return position_features(verts, nb_mask)
    if kind == 'random':
        rng = np.random.default_rng(999)            # 全seedで固定（頂点ごとの定数）
        return rng.standard_normal((V, 2)).astype(np.float32)
    raise ValueError(kind)


def run_once(features, nb_idx, nb_ang, nb_mask, aux, n_vars, args, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    V = nb_idx.shape[0]
    T = len(features)
    T_tr, T_va = int(T * 0.7), int(T * 0.15)

    tr = features[:T_tr]
    mean = tr.mean(axis=(0, 1), keepdims=True)
    std  = tr.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    feats = (features - mean) / std

    ds_tr = MeshForecastDataset(feats[:T_tr],           lag=args.lag, normalize=False)
    ds_va = MeshForecastDataset(feats[T_tr:T_tr+T_va],  lag=args.lag, normalize=False)
    ds_te = MeshForecastDataset(feats[T_tr+T_va:],      lag=args.lag, normalize=False)

    use_aux = aux is not None
    c_in = n_vars + (2 if use_aux else 0)

    def cat_aux(X):
        if not use_aux:
            return X
        ab = torch.tensor(aux, dtype=torch.float32)[None].expand(len(X), V, 2)
        return torch.cat([X, ab], dim=-1)

    ds_tr.X = cat_aux(ds_tr.X)
    ds_va.X = cat_aux(ds_va.X)
    te_X    = cat_aux(ds_te.X)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    model = IcosGaugeCNNGeneral(c_in=c_in, c_hidden=args.c_hidden, n_types=args.n_types,
                                n_out=n_vars, nb_idx=nb_idx, nb_ang=nb_ang, nb_mask=nb_mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()

    best_val, best_state = float('inf'), None
    for _ in range(args.epochs):
        train_one_epoch(model, dl_tr, opt, crit, device)
        vl = evaluate(model, dl_va, crit, device)
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mse = crit(model(te_X.to(device)), ds_te.y.to(device)).item()
    skill = 1.0 - mse / _persistence_loss(ds_te, crit, device)
    return mse, skill


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--level',   type=int, default=1)
    p.add_argument('--T',       type=int, default=300)
    p.add_argument('--lag',     type=int, default=4)
    p.add_argument('--c_hidden',type=int, default=16)
    p.add_argument('--n_types', type=int, default=4)
    p.add_argument('--epochs',  type=int, default=60)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--batch',   type=int, default=16)
    p.add_argument('--seeds',   type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    verts0, faces0 = build_icosahedron()[:2]
    verts, faces = subdivide(verts0, faces0, level=args.level)
    V = len(verts)
    nb_idx, nb_ang, nb_mask, verts, faces = build_gauge_data_general(verts, faces, device)
    nb_mask_np = nb_mask.cpu().numpy()

    n_vars = 3
    np.random.seed(12345)
    features = make_synthetic_era5(verts, T=args.T, n_vars=n_vars,
                                   nb_idx=nb_idx.cpu().numpy())
    print(f"Icosahedron level {args.level}: {V}頂点  合成ERA5 {features.shape}\n")

    groups = ['baseline', 'random', 'position', 'holonomy']
    aux = {g: make_aux(g, verts, faces, nb_mask_np, V) for g in groups}
    res = {g: [] for g in groups}

    for seed in args.seeds:
        line = [f"seed {seed}:"]
        for g in groups:
            mse, sk = run_once(features, nb_idx, nb_ang, nb_mask, aux[g],
                               n_vars, args, device, seed)
            res[g].append((mse, sk))
            line.append(f"{g[:4]} {mse:.3f}/{sk:+.2f}")
        print("  ".join(line))

    print(f"\n{'='*60}")
    print(f"  {'group':10s} {'MSE':>16s}   {'skill':>7s}")
    base_mse = np.array(res['baseline'])[:, 0].mean()
    for g in groups:
        a = np.array(res[g])
        m, s, k = a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()
        imp = 100 * (base_mse - m) / base_mse
        print(f"  {g:10s} {m:.4f} ± {s:.4f}   {k:+.4f}   ({imp:+.1f}% vs base)")
    print(f"{'='*60}")
    mh = np.array(res['holonomy'])[:, 0].mean()
    mp = np.array(res['position'])[:, 0].mean()
    mr = np.array(res['random'])[:, 0].mean()
    print(f"\n  解釈:")
    print(f"   random vs baseline   : {100*(base_mse-mr)/base_mse:+.1f}%  (ただの2列増の効果)")
    print(f"   position vs baseline : {100*(base_mse-mp)/base_mse:+.1f}%  (位置ヒントの効果)")
    print(f"   holonomy vs position : {100*(mp-mh)/mp:+.1f}%  ← 曲率特有の寄与")


if __name__ == '__main__':
    main()
