"""
ERA5 × IcosGaugeCNN — 球面気温予測タスク

正二十面体メッシュ（subdivision level 指定可）上で
IcosGaugeCNNGeneral を使い、t 時刻の特徴量から t+lag 時刻を予測する。

使い方:
  python era5_task.py                     # 合成データで動作確認
  python era5_task.py --nc era5.nc        # ERA5 の実データ
  python era5_task.py --level 2           # level 2 (162 頂点)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from icosahedron import build_icosahedron, subdivide, build_gauge_data_general
from gauge_cnn import IcosGaugeCNNGeneral
from era5_loader import make_synthetic_era5, load_era5_nc, load_weatherbench2


# ============================================================
# データセット
# ============================================================

class MeshForecastDataset(Dataset):
    """
    (features[t], features[t+lag]) のペアを返すデータセット。

    Args:
        features : (T, V, C)  時系列特徴量
        lag      : 予測ラグ（タイムステップ数）
        normalize: 変数ごとに標準化するか
    """

    def __init__(self, features: np.ndarray, lag: int = 1, normalize: bool = True):
        T, V, C = features.shape
        self.lag = lag

        if normalize:
            mean = features.mean(axis=(0, 1), keepdims=True)  # (1,1,C)
            std  = features.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
            features = (features - mean) / std
            self.mean = mean.squeeze()   # (C,)
            self.std  = std.squeeze()
        else:
            self.mean = None
            self.std  = None

        self.X = torch.tensor(features[:T - lag], dtype=torch.float32)  # (T-lag, V, C)
        self.y = torch.tensor(features[lag:],     dtype=torch.float32)  # (T-lag, V, C)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 学習ループ
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)        # (B, V, n_out)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item()
    return total_loss / len(loader)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level',   type=int, default=1,
                        help='Icosahedron subdivision level (0=12, 1=42, 2=162, 3=642)')
    parser.add_argument('--nc',      type=str, default=None,
                        help='ERA5 NetCDF ファイルパス')
    parser.add_argument('--wb2',     action='store_true',
                        help='WeatherBench2 (GCS) からデータを取得する')
    parser.add_argument('--wb2_start', type=str, default='2018-01-01')
    parser.add_argument('--wb2_end',   type=str, default='2019-12-31')
    parser.add_argument('--vars',    nargs='+', default=['t2m'],
                        help='ERA5 変数名（--nc 指定時）')
    parser.add_argument('--lag',     type=int, default=4,
                        help='予測ラグ（タイムステップ数、ERA5 6時間ごとなら lag=4 で 24h 予測）')
    parser.add_argument('--T',       type=int, default=200,
                        help='合成データのタイムステップ数')
    parser.add_argument('--c_hidden',type=int, default=16)
    parser.add_argument('--n_types', type=int, default=4)
    parser.add_argument('--epochs',  type=int, default=50)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--batch',   type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── メッシュ構築 ────────────────────────────────────────────────────
    verts0, faces0 = build_icosahedron()[:2]
    verts, faces   = subdivide(verts0, faces0, level=args.level)
    V = len(verts)
    print(f"Icosahedron level {args.level}: {V} 頂点")

    nb_idx, nb_ang, nb_mask, verts, faces = build_gauge_data_general(verts, faces, device)
    max_deg = nb_idx.shape[1]
    print(f"  max_deg={max_deg}")

    # ── データロード ────────────────────────────────────────────────────
    if args.wb2:
        features, times = load_weatherbench2(
            verts, variables=args.vars,
            start=args.wb2_start, end=args.wb2_end,
        )
        n_vars = len(args.vars)
    elif args.nc is not None:
        print(f"ERA5 ロード中: {args.nc}")
        features, times = load_era5_nc(args.nc, args.vars, verts)
        print(f"  shape={features.shape}  period={times[0]} → {times[-1]}")
        n_vars = len(args.vars)
    else:
        print(f"合成データ生成中 (T={args.T}, level={args.level}, 空間拡散あり)...")
        n_vars   = 3
        nb_idx_np = nb_idx.cpu().numpy()
        features = make_synthetic_era5(verts, T=args.T, n_vars=n_vars, nb_idx=nb_idx_np)
        print(f"  shape={features.shape}")

    # ── Train / Val / Test 分割（時系列なので前後で）──────────────────
    T_total = len(features)
    T_train = int(T_total * 0.7)
    T_val   = int(T_total * 0.15)

    # train セットの統計でまとめて正規化（リーク防止）
    train_feat = features[:T_train]
    mean = train_feat.mean(axis=(0, 1), keepdims=True)
    std  = train_feat.std(axis=(0, 1),  keepdims=True).clip(min=1e-6)
    features = (features - mean) / std

    ds_train = MeshForecastDataset(features[:T_train],             lag=args.lag, normalize=False)
    ds_val   = MeshForecastDataset(features[T_train:T_train+T_val],lag=args.lag, normalize=False)
    ds_test  = MeshForecastDataset(features[T_train+T_val:],       lag=args.lag, normalize=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False)
    print(f"Train: {len(ds_train)}  Val: {len(ds_val)}  Test: {len(ds_test)}")

    # ── モデル ──────────────────────────────────────────────────────────
    model = IcosGaugeCNNGeneral(
        c_in     = n_vars,
        c_hidden = args.c_hidden,
        n_types  = args.n_types,
        n_out    = n_vars,
        nb_idx   = nb_idx,
        nb_ang   = nb_ang,
        nb_mask  = nb_mask,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    # ── 学習 ────────────────────────────────────────────────────────────
    best_val = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, device)
        val_loss   = evaluate(model, dl_val, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4}/{args.epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"(best={best_val:.4f})")

    # ── テスト評価 ──────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    test_loss = evaluate(model, dl_test, criterion, device)
    # Skill score: MSE vs. 持続予測（前時刻をそのまま使う）
    persist_loss = _persistence_loss(ds_test, criterion, device)
    skill = 1.0 - test_loss / persist_loss

    print(f"\n── テスト結果 ──")
    print(f"  MSE:             {test_loss:.4f}")
    print(f"  持続予測 MSE:    {persist_loss:.4f}")
    print(f"  Skill score:     {skill:.4f}  (>0 で持続予測より良い)")


@torch.no_grad()
def _persistence_loss(ds, criterion, device):
    """持続予測（X をそのまま y とする）の MSE。"""
    total = 0.0
    for X, y in DataLoader(ds, batch_size=32, shuffle=False):
        total += criterion(X.to(device), y.to(device)).item()
    return total / len(DataLoader(ds, batch_size=32))


if __name__ == '__main__':
    main()
