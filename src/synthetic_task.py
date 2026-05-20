"""
合成タスク: SphericalCNN v1 vs v2 の表現力比較

─────────────────────────────────────────────────────────────────
v1 (NormNonlinearity) の根本的制限
─────────────────────────────────────────────────────────────────
S2EquivariantLinear は ℓ ブロックを独立に処理（Schurの補題）。
NormNonlinearity は ℓ 内のノルムだけをスケール。
readout は ℓ=0 スロットのみ取り出す。

→ ℓ=0 スロットには常に入力の ℓ=0 成分だけが流れてくる。
  ℓ≥1 の情報は絶対に readout に届かない。

─────────────────────────────────────────────────────────────────
2 つのタスクで実証
─────────────────────────────────────────────────────────────────
タスク A (easy): y = f̂_{0,0}²    ← ℓ=0 パワー
  → v1 は学習できる（ℓ=0 情報のみ必要）
  → v2 も学習できる

タスク B (hard): y = Σ_m f̂_{1,m}²  ← ℓ=1 パワー (SO(3) 不変)
  → v1 は学習不可能（ℓ=1 が readout に届かない）
  → v2 は学習できる（CG 積 ℓ=1⊗ℓ=1→ℓ=0 でスカラーを生成）

理論的には: v2 で 2 層使えば ℓ=1⊗ℓ=1=ℓ=0 の CG パスが直接
  Σ_m f̂_{1,m}² を ℓ=0 に集約できる。
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from spherical_cnn import (
    SphericalHarmonicTransform, SphericalCNN, SphericalCNNv2,
)

# ─── 設定 ──────────────────────────────────────────────────────
L_MAX   = 4       # 小さく保って高速化（v1 制限は L_max に依存しない）
N_THETA = 16
N_PHI   = 32
N_TRAIN = 3000
N_VAL   = 500
EPOCHS  = 150
LR      = 3e-3
BATCH   = 256
CHANNELS = [1, 8, 8]  # v1/v2 共通

# ─── データ生成 ────────────────────────────────────────────────

def generate_dataset(sht: SphericalHarmonicTransform,
                     n: int, seed: int = 0):
    """
    バンド制限球面信号を生成。

    SH 係数: c_{l,m} ~ N(0, σ_l²)
      σ_l = 1.0 (l ≤ 2), 0.1 (l > 2)

    返却:
      f_spatial: (n, N)    球面グリッド上の信号
      y_a:       (n, 1)    タスクA: f̂_{0,0}²
      y_b:       (n, 1)    タスクB: Σ_m f̂_{1,m}²
    """
    rng = np.random.default_rng(seed)
    L2 = L_MAX ** 2

    # SH 係数
    c = np.zeros((n, L2), dtype=np.float32)
    for l in range(L_MAX):
        sigma = 1.0 if l <= 2 else 0.1
        for m in range(-l, l + 1):
            c[:, l**2 + l + m] = rng.normal(0, sigma, n)

    # 空間信号: f = c @ Y_synthesis^T
    Y_s = sht.Y_synthesis.astype(np.float32)   # (N, L²)
    f_spatial = c @ Y_s.T                        # (n, N)

    # タスク A: f̂_{0,0}²  (index 0)
    y_a = c[:, 0:1] ** 2

    # タスク B: Σ_m f̂_{1,m}²  (ℓ=1 の 3 成分: indices 1,2,3)
    y_b = (c[:, 1:4] ** 2).sum(axis=1, keepdims=True)

    return f_spatial, y_a, y_b


# ─── 学習ユーティリティ ────────────────────────────────────────

def train_model(model: nn.Module, f_train, y_train, f_val, y_val,
                task_label: str, model_label: str) -> tuple[list, list]:
    """MSE 回帰を学習。train/val loss の履歴を返す。"""
    ds_train = TensorDataset(
        torch.tensor(f_train), torch.tensor(y_train))
    ds_val   = TensorDataset(
        torch.tensor(f_val),   torch.tensor(y_val))
    loader_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_hist, val_hist = [], []
    best_val = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader_train:
            opt.zero_grad()
            nn.MSELoss()(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(
                model(torch.tensor(f_train)),
                torch.tensor(y_train)).item()
            val_loss   = criterion(
                model(torch.tensor(f_val)),
                torch.tensor(y_val)).item()
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss

        if epoch % 30 == 0 or epoch == 1:
            print(f"  [{model_label}/{task_label}] "
                  f"epoch {epoch:3d}/{EPOCHS} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

    return train_hist, val_hist, best_val


# ─── メイン ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"合成タスク: SphericalCNN v1 vs v2")
    print(f"  L_max={L_MAX}, grid=({N_THETA}×{N_PHI})")
    print(f"  channels={CHANNELS}, epochs={EPOCHS}")
    print("=" * 60)

    sht = SphericalHarmonicTransform(L_MAX, N_THETA, N_PHI)

    print("\nデータ生成中...")
    f_train, ya_train, yb_train = generate_dataset(sht, N_TRAIN, seed=0)
    f_val,   ya_val,   yb_val   = generate_dataset(sht, N_VAL,   seed=1)

    # ベースライン: 定数予測のMSE = Var(y)
    var_a = float(np.var(ya_train))
    var_b = float(np.var(yb_train))
    print(f"\nVar(y_A) = {var_a:.4f}  (定数予測の MSE)")
    print(f"Var(y_B) = {var_b:.4f}  (定数予測の MSE)")

    results = {}

    # ── タスク A ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("タスク A: y = f̂₀₀²  (ℓ=0 パワー) — v1 が学習できるはず")
    print("─" * 60)

    print("\n[v1 学習中]")
    v1_a = SphericalCNN(sht, channels=CHANNELS, n_layers=2)
    _, _, best_a_v1 = train_model(v1_a, f_train, ya_train, f_val, ya_val,
                                   "A", "v1")

    print("\n[v2 学習中]")
    v2_a = SphericalCNNv2(sht, channels=CHANNELS, n_layers=2)
    _, _, best_a_v2 = train_model(v2_a, f_train, ya_train, f_val, ya_val,
                                   "A", "v2")
    results['A'] = {'v1': best_a_v1, 'v2': best_a_v2, 'var': var_a}

    # ── タスク B ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("タスク B: y = Σ_m f̂₁ₘ²  (ℓ=1 パワー) — v1 は学習不可のはず")
    print("─" * 60)

    print("\n[v1 学習中]")
    v1_b = SphericalCNN(sht, channels=CHANNELS, n_layers=2)
    _, _, best_b_v1 = train_model(v1_b, f_train, yb_train, f_val, yb_val,
                                   "B", "v1")

    print("\n[v2 学習中]")
    v2_b = SphericalCNNv2(sht, channels=CHANNELS, n_layers=2)
    _, _, best_b_v2 = train_model(v2_b, f_train, yb_train, f_val, yb_val,
                                   "B", "v2")
    results['B'] = {'v1': best_b_v1, 'v2': best_b_v2, 'var': var_b}

    # ── 結果サマリー ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    for task, name in [("A", "y = f̂₀₀²  (ℓ=0 パワー)"),
                        ("B", "y = Σ f̂₁ₘ²  (ℓ=1 パワー)")]:
        r = results[task]
        v1_rel = r['v1'] / r['var']
        v2_rel = r['v2'] / r['var']
        print(f"\nタスク {task}: {name}")
        print(f"  Var(y) = {r['var']:.4f}  (定数予測の MSE = 1.00)")
        print(f"  v1 best val MSE = {r['v1']:.4f}  (相対 {v1_rel:.3f})")
        print(f"  v2 best val MSE = {r['v2']:.4f}  (相対 {v2_rel:.3f})")
        if task == "B":
            if v2_rel < 0.1 and v1_rel > 0.8:
                verdict = "✓ v2 が学習、v1 が失敗 — CG 層の表現力差を確認"
            elif v2_rel < v1_rel * 0.5:
                verdict = "✓ v2 が v1 を大幅に上回る"
            else:
                verdict = "? 差が小さい — エポック数や設定を見直すかも"
            print(f"  → {verdict}")
    print("=" * 60)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f"\n総実行時間: {time.time()-start:.1f}s")
