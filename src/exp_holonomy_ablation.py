"""
曲率特徴量（Wilsonプラケットのホロノミー）のablation実験

格子ゲージ理論（Harlow QFT3 §8）由来のゲージ不変な曲率特徴を
GaugeCNN の入力に足すと、ERA5予測タスクの精度が上がるかを検証する。

同一データ・同一seed・同一ハイパラで、入力に曲率特徴(2列)を
concatするかどうかだけを変えて比較する（公平なablation）。

使い方:
  python exp_holonomy_ablation.py                 # 合成データ, level 1
  python exp_holonomy_ablation.py --level 2 --seeds 0 1 2
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
from era5_loader import make_synthetic_era5
from era5_task import MeshForecastDataset, train_one_epoch, evaluate, _persistence_loss


def holonomy_vertex_features(verts, faces):
    """subdivision後のメッシュの頂点ごとのゲージ不変曲率特徴 (V,2)。"""
    edge_set = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([int(f[i]), int(f[(i + 1) % 3])]))
            edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int64)
    frames = compute_local_frames(verts)
    angles = compute_connection_angles(verts, edges, frames)
    holo = compute_plaquette_holonomy(faces, edges, angles)
    return scatter_holonomy_to_vertices(faces, holo, len(verts))   # (V,2)


def run_once(features, nb_idx, nb_ang, nb_mask, holo_feat,
             use_holonomy, n_vars, args, device, seed):
    """1回の学習→テストを実行し test MSE と skill を返す。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    V = nb_idx.shape[0]
    T_total = len(features)
    T_train = int(T_total * 0.7)
    T_val   = int(T_total * 0.15)

    # train統計で正規化（リーク防止）
    train_feat = features[:T_train]
    mean = train_feat.mean(axis=(0, 1), keepdims=True)
    std  = train_feat.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    feats = (features - mean) / std

    ds_train = MeshForecastDataset(feats[:T_train],               lag=args.lag, normalize=False)
    ds_val   = MeshForecastDataset(feats[T_train:T_train+T_val],  lag=args.lag, normalize=False)
    ds_test  = MeshForecastDataset(feats[T_train+T_val:],         lag=args.lag, normalize=False)

    # 曲率特徴を入力に concat する版
    # （ターゲット y は気象変数のみ。曲率は入力補助なので n_out は n_vars のまま）
    c_in = n_vars + (2 if use_holonomy else 0)

    def add_holo(ds):
        if not use_holonomy:
            return ds
        # X に曲率2列を足した新データセットを作る（y は元のまま）
        hb = torch.tensor(holo_feat, dtype=torch.float32)[None].expand(len(ds.X), V, 2)
        ds.X = torch.cat([ds.X, hb], dim=-1)
        return ds

    ds_train = add_holo(ds_train)
    ds_val   = add_holo(ds_val)
    # test の X も曲率concat（y_test は気象のみ）
    if use_holonomy:
        hb = torch.tensor(holo_feat, dtype=torch.float32)[None].expand(len(ds_test.X), V, 2)
        ds_test_X = torch.cat([ds_test.X, hb], dim=-1)
    else:
        ds_test_X = ds_test.X

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False)

    model = IcosGaugeCNNGeneral(
        c_in=c_in, c_hidden=args.c_hidden, n_types=args.n_types,
        n_out=n_vars, nb_idx=nb_idx, nb_ang=nb_ang, nb_mask=nb_mask,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()

    best_val, best_state = float('inf'), None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, dl_train, opt, crit, device)
        vl = evaluate(model, dl_val, crit, device)
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # test 評価（手動で X を渡す）
    model.eval()
    with torch.no_grad():
        Xb = ds_test_X.to(device)
        yb = ds_test.y.to(device)
        test_mse = crit(model(Xb), yb).item()
    persist = _persistence_loss(ds_test, crit, device)
    skill = 1.0 - test_mse / persist
    return test_mse, skill, n_params


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
    p.add_argument('--seeds',   type=int, nargs='+', default=[0, 1, 2])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    verts0, faces0 = build_icosahedron()[:2]
    verts, faces   = subdivide(verts0, faces0, level=args.level)
    V = len(verts)
    nb_idx, nb_ang, nb_mask, verts, faces = build_gauge_data_general(verts, faces, device)
    holo_feat = holonomy_vertex_features(verts, faces)   # (V,2)
    print(f"Icosahedron level {args.level}: {V}頂点  曲率特徴の多様性: "
          f"{len(np.unique(holo_feat[:,0].round(4)))}種")

    # 共通の合成データ（seedごとに作り直さず固定→入力以外の条件を完全一致させる）
    nb_idx_np = nb_idx.cpu().numpy()
    n_vars = 3
    np.random.seed(12345)
    features = make_synthetic_era5(verts, T=args.T, n_vars=n_vars, nb_idx=nb_idx_np)
    print(f"合成ERA5: {features.shape}\n")

    results = {'baseline': [], 'holonomy': []}
    for seed in args.seeds:
        mse_b, sk_b, np_b = run_once(features, nb_idx, nb_ang, nb_mask, holo_feat,
                                     False, n_vars, args, device, seed)
        mse_h, sk_h, np_h = run_once(features, nb_idx, nb_ang, nb_mask, holo_feat,
                                     True, n_vars, args, device, seed)
        results['baseline'].append((mse_b, sk_b))
        results['holonomy'].append((mse_h, sk_h))
        print(f"seed {seed}:  baseline MSE={mse_b:.4f} skill={sk_b:+.4f}  | "
              f"holonomy MSE={mse_h:.4f} skill={sk_h:+.4f}  | "
              f"ΔMSE={mse_h-mse_b:+.4f}")

    def agg(key):
        a = np.array(results[key])
        return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()
    mb, sb, kb = agg('baseline')
    mh, sh, kh = agg('holonomy')
    print(f"\n{'='*56}")
    print(f"  baseline : MSE {mb:.4f} ± {sb:.4f}   skill {kb:+.4f}")
    print(f"  holonomy : MSE {mh:.4f} ± {sh:.4f}   skill {kh:+.4f}")
    print(f"  改善      : ΔMSE {mh-mb:+.4f}  ({100*(mb-mh)/mb:+.2f}%)  "
          f"(パラメータ増 {np_h-np_b})")
    print(f"{'='*56}")


if __name__ == '__main__':
    main()
