"""
非自明なU(1)ゲージ場（本物の曲率場）での曲率特徴の検証

これまでの合成データは曲率が trivial（メッシュ幾何のみ、全頂点ほぼ平坦）で、
曲率特徴の物理的価値が出なかった。そこで:

  1. 各辺にランダムな追加接続 β_e を注入 → プラケットに非自明なホロノミー（曲率場）
  2. ターゲット・ダイナミクスを局所曲率で変調する
     （曲率の大きい頂点ほど時間発展が速い＝曲率依存の拡散）
  → 曲率を知らないと原理的に予測できないタスクになる。

4群比較（baseline / random / position / holonomy）。
前回(exp_holonomy_3way)は曲率trivialで holonomy≈random だった。
今回 holonomy >> random なら「曲率特徴は本物の曲率場では効く」と言える。

使い方:
  python exp_holonomy_nontrivial.py --seeds 0 1 2 3 4
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from icosahedron import (build_icosahedron, subdivide, build_gauge_data_general,
                         compute_local_frames, compute_connection_angles,
                         compute_plaquette_holonomy, scatter_holonomy_to_vertices)
from gauge_cnn import IcosGaugeCNNGeneral
from era5_loader import xyz_to_latlon
from era5_task import MeshForecastDataset, train_one_epoch, evaluate, _persistence_loss


def build_edges(faces):
    s = set()
    for f in faces:
        for i in range(3):
            s.add(tuple(sorted([int(f[i]), int(f[(i + 1) % 3])])))
    return np.array(sorted(s), dtype=np.int64)


def make_nontrivial_gauge(verts, faces, edges, gauge_seed=7, strength=1.0):
    """各辺に追加接続β_eを注入した非自明ゲージ場のホロノミー(面)と頂点曲率(V,2)を返す。"""
    frames = compute_local_frames(verts)
    angles = compute_connection_angles(verts, edges, frames)
    rng = np.random.default_rng(gauge_seed)
    beta = rng.uniform(-strength, strength, size=len(edges))
    angles = angles.copy()
    angles[:, 0] += beta
    angles[:, 1] -= beta
    holo = compute_plaquette_holonomy(faces, edges, angles)         # (F,)
    vert_curv = scatter_holonomy_to_vertices(faces, holo, len(verts))  # (V,2)
    return holo, vert_curv, angles


def make_curvature_modulated_data(verts, faces, edges, nb_idx_np, holo_face,
                                  T=300, n_vars=3, data_seed=12345):
    """
    曲率が「本質的に必要」な予測タスクを作る。

    設計の核心:
      入力場 x は全頂点で同じ統計の共通ダイナミクス（曲率に依存しない）。
      ターゲット y を曲率の非自明な関数で変調する:
          y_v[t] = cos(Θ_v) · x_v[t+lag] + sin(Θ_v) · (隣接平均)_v[t+lag]
      ここで Θ_v は頂点の局所ホロノミー（曲率）。

    なぜ曲率が必須か:
      - x の時系列統計は全頂点で同じ（左右対称）→ 時系列だけでは cos/sin(Θ_v) を
        復元できない。
      - ランダム特徴は Θ_v と無相関 → y を決める情報を持たない。
      - 曲率特徴 [cosΘ̄_v, sinΘ̄_v] だけが変調係数を直接与える。
      ⇒ symmetry breaking では解けず、曲率の中身が必要なタスク。

    ※ この関数は (features, targets) を返す（入力とターゲットが別物）。
    """
    V = len(verts)
    # 変調係数は holonomy特徴 [cosΘ̄_v, sinΘ̄_v] と完全一致させる
    # （これにより「曲率特徴が変調係数そのもの」になり、曲率の寄与をクリーンに測れる。
    #   ランダム特徴は無相関なので解けない＝symmetry breakingと切り分けられる）
    vc = scatter_holonomy_to_vertices(faces, holo_face, V)   # (V,2)=[cosΘ̄, sinΘ̄]
    cz, sz = vc[:, 0], vc[:, 1]

    # 入力場 x: 全頂点共通統計の空間拡散AR(1)（曲率に依存しない）
    rng = np.random.default_rng(data_seed)
    x = np.zeros((T, V, n_vars), dtype=np.float32)
    alpha, phi = 0.4, 0.85
    for c in range(n_vars):
        state = rng.standard_normal(V)
        for t in range(T):
            noise = rng.standard_normal(V) * 0.3
            nbvals = state[nb_idx_np]
            diffused = alpha * nbvals.mean(axis=1) + (1 - alpha) * state
            state = phi * diffused + noise
            x[t, :, c] = state.astype(np.float32)

    # ターゲット y: 曲率で変調（x自身と隣接平均を曲率係数で混ぜる）
    y = np.zeros_like(x)
    for c in range(n_vars):
        nb_mean = x[:, :, c][:, nb_idx_np].mean(axis=2)   # (T,V)
        y[:, :, c] = cz[None, :] * x[:, :, c] + sz[None, :] * nb_mean
    return x, y


def position_features(verts, nb_mask):
    deg = nb_mask.sum(axis=1)
    lats, _ = xyz_to_latlon(verts)
    def z(a):
        a = a.astype(np.float32); return (a - a.mean()) / (a.std() + 1e-6)
    return np.stack([z(deg), z(lats)], axis=1).astype(np.float32)


def make_aux(kind, vert_curv, verts, nb_mask_np, V):
    if kind == 'baseline': return None
    if kind == 'holonomy': return vert_curv
    if kind == 'position': return position_features(verts, nb_mask_np)
    if kind == 'random':
        return np.random.default_rng(999).standard_normal((V, 2)).astype(np.float32)
    raise ValueError(kind)


class XYDataset(torch.utils.data.Dataset):
    """入力X→ターゲットYの直接マッピング用（時系列ラグではない）。"""
    def __init__(self, X, Y):
        self.X = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)
        self.Y = Y if torch.is_tensor(Y) else torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def run_once(x, y, nb_idx, nb_ang, nb_mask, aux, n_vars, args, device, seed,
             aux_test=None, y_test=None, x_test=None):
    """
    aux_test/y_test/x_test が与えられたら、テストは別ゲージ場で評価する
    （train/test で曲率場が異なる＝暗記不能、曲率の汎化を強制する設計）。
    """
    torch.manual_seed(seed); np.random.seed(seed)
    V = nb_idx.shape[0]; T = len(x)
    T_tr, T_va = int(T * 0.7), int(T * 0.15)

    # 入力xをtrain統計で正規化
    xm = x[:T_tr].mean(axis=(0, 1), keepdims=True)
    xs = x[:T_tr].std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    xn = (x - xm) / xs

    use = aux is not None
    c_in = n_vars + (2 if use else 0)
    def cat(Xnp, a):
        Xt = torch.tensor(Xnp, dtype=torch.float32)
        if not use: return Xt
        ab = torch.tensor(a, dtype=torch.float32)[None].expand(len(Xt), V, 2)
        return torch.cat([Xt, ab], dim=-1)

    Xtr, Xva = cat(xn[:T_tr], aux), cat(xn[T_tr:T_tr+T_va], aux)
    Ytr = torch.tensor(y[:T_tr], dtype=torch.float32)
    Yva = torch.tensor(y[T_tr:T_tr+T_va], dtype=torch.float32)

    if y_test is not None:
        # 別ゲージ場のテストセット（同じ入力正規化統計を流用）
        xnt = (x_test - xm) / xs
        a_te = aux_test if use else None
        Xte = cat(xnt[T_tr+T_va:], a_te)
        Yte = torch.tensor(y_test[T_tr+T_va:], dtype=torch.float32)
    else:
        Xte = cat(xn[T_tr+T_va:], aux)
        Yte = torch.tensor(y[T_tr+T_va:], dtype=torch.float32)

    dl_tr = DataLoader(XYDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True)
    dl_va = DataLoader(XYDataset(Xva, Yva), batch_size=args.batch, shuffle=False)

    model = IcosGaugeCNNGeneral(c_in=c_in, c_hidden=args.c_hidden, n_types=args.n_types,
                                n_out=n_vars, nb_idx=nb_idx, nb_ang=nb_ang, nb_mask=nb_mask).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    best, best_state = float('inf'), None
    for _ in range(args.epochs):
        train_one_epoch(model, dl_tr, opt, crit, device)
        vl = evaluate(model, dl_va, crit, device); sched.step(vl)
        if vl < best: best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mse = crit(model(Xte.to(device)), Yte.to(device)).item()
    # ベースライン: y の分散（恒等予測=平均との比較）
    var_y = Yte.var().item()
    return mse, 1.0 - mse / var_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--level', type=int, default=1)
    p.add_argument('--T', type=int, default=300)
    p.add_argument('--lag', type=int, default=4)
    p.add_argument('--c_hidden', type=int, default=16)
    p.add_argument('--n_types', type=int, default=4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--gauge_strength', type=float, default=1.0)
    p.add_argument('--split_gauge', action='store_true',
                   help='train と test で異なるゲージ場を使う（暗記不能＝曲率の汎化を強制）')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    verts0, faces0 = build_icosahedron()[:2]
    verts, faces = subdivide(verts0, faces0, level=args.level)
    V = len(verts)
    nb_idx, nb_ang, nb_mask, verts, faces = build_gauge_data_general(verts, faces, device)
    nb_mask_np = nb_mask.cpu().numpy(); nb_idx_np = nb_idx.cpu().numpy()
    edges = build_edges(faces)

    # 非自明ゲージ場 + 曲率変調ダイナミクス（train用）
    holo_face, vert_curv, _ = make_nontrivial_gauge(verts, faces, edges,
                                                    strength=args.gauge_strength,
                                                    gauge_seed=7)
    print(f"非自明ゲージ場(train): 曲率std={np.degrees(holo_face.std()):.1f}°  "
          f"頂点曲率多様性={len(np.unique(vert_curv[:,0].round(3)))}種")
    x, y = make_curvature_modulated_data(verts, faces, edges, nb_idx_np,
                                         holo_face, T=args.T)

    # split_gauge: テストは別のゲージ場（暗記不能、曲率の汎化を強制）
    if args.split_gauge:
        holo_te, vert_curv_te, _ = make_nontrivial_gauge(verts, faces, edges,
                                                         strength=args.gauge_strength,
                                                         gauge_seed=999)
        x_te, y_te = make_curvature_modulated_data(verts, faces, edges, nb_idx_np,
                                                   holo_te, T=args.T, data_seed=54321)
        print("★split_gauge: test は別ゲージ場(seed999)。暗記不能＝曲率の汎化を測る")
    else:
        x_te = y_te = vert_curv_te = None
    print(f"曲率変調タスク: 入力{x.shape} → ターゲット{y.shape}（y=曲率変調）\n")

    groups = ['baseline', 'random', 'position', 'holonomy']
    aux = {g: make_aux(g, vert_curv, verts, nb_mask_np, V) for g in groups}
    # テスト用aux（split時はholonomyだけ別ゲージ場の曲率に差し替え。他は同じ）
    aux_te = {}
    for g in groups:
        if not args.split_gauge:
            aux_te[g] = None
        elif g == 'holonomy':
            aux_te[g] = vert_curv_te        # テスト時は新ゲージ場の曲率を入力
        else:
            aux_te[g] = aux[g]              # baseline/random/position は変わらず
    res = {g: [] for g in groups}
    for seed in args.seeds:
        line = [f"seed {seed}:"]
        for g in groups:
            mse, sk = run_once(x, y, nb_idx, nb_ang, nb_mask, aux[g], 3, args, device, seed,
                               aux_test=aux_te[g], y_test=y_te, x_test=x_te)
            res[g].append((mse, sk)); line.append(f"{g[:4]} {mse:.3f}/{sk:+.2f}")
        print("  ".join(line))

    print(f"\n{'='*60}")
    base = np.array(res['baseline'])[:, 0].mean()
    for g in groups:
        a = np.array(res[g]); m, s, k = a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()
        print(f"  {g:10s} {m:.4f} ± {s:.4f}   skill {k:+.4f}   ({100*(base-m)/base:+.1f}%)")
    print(f"{'='*60}")
    mh = np.array(res['holonomy'])[:, 0].mean()
    mr = np.array(res['random'])[:, 0].mean()
    mp = np.array(res['position'])[:, 0].mean()
    print(f"\n  ★曲率特有の寄与（vs random）: {100*(mr-mh)/mr:+.1f}%")
    print(f"   曲率 vs position           : {100*(mp-mh)/mp:+.1f}%")
    print(f"   （前回 trivial 曲率では vs random +0.7%。今回 >> なら曲率場で効く証拠）")


if __name__ == '__main__':
    main()
