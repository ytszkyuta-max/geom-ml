"""
球面CNN 同変性テスト

テスト戦略:
  z軸回転 (φ → φ + φ_0) はグリッドと整合させれば np.roll で厳密に実現できる。
  φ_0 = 2π·k/N_PHI とすると補間誤差がゼロになり、数値誤差の源が SHT の
  離散化精度のみに絞られる。

  T1: SHT 可逆性             — バンド制限信号で forward○backward が恒等
  T2: SphericalCNN 回転不変性 — model(R·f) ≈ model(f)
       ℓ=0 成分のみ取り出すアーキテクチャなので出力は SO(3) 不変スカラーのはず
  T3: S2EquivariantLinear 回転同変性
       iSHT( layer( SHT(R·f) ) ) ≈ R · iSHT( layer( SHT(f) ) )
       "先に回転してから処理" = "処理してから回転" が成立するか検証
  T4: ClebschGordanProduct 回転同変性（スペクトル域）
       実球面調和基底での Wigner D 行列（2×2 回転ブロック）を使って
       スペクトル域で直接回転を適用し CG 層の同変性を検証する
       空間域を経由しないので SHT 求積誤差が混入しない
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from spherical_cnn import (SphericalHarmonicTransform, S2EquivariantLinear,
                          SphericalCNN, ClebschGordanProduct, SphericalCNNv2,
                          wigner_d_real)


# ─── 定数 ────────────────────────────────────────────────────────────────────

L_MAX   = 6
N_THETA = 32
N_PHI   = 64   # 整数 k で割り切れる値を使う
BATCH   = 3
# スペクトル往復 (forward ∘ backward): Y_w^T · Y ≈ I の直交性誤差
# 均等グリッドの求積誤差により ~6e-4 程度の残差が生じる。max ~1.3e-3 なので 5e-3 で余裕を持つ
TOL_SHT   = 5e-3
TOL_EQUIV = 5e-3


# ─── ヘルパー ─────────────────────────────────────────────────────────────────

def make_sht() -> SphericalHarmonicTransform:
    return SphericalHarmonicTransform(L_MAX, N_THETA, N_PHI)


def band_limited_signal(sht: SphericalHarmonicTransform,
                        rng: np.random.Generator) -> np.ndarray:
    """
    バンド制限信号を生成。
    ランダムスペクトル係数から iSHT で合成するので L_max 以下に
    厳密にバンド制限されており、SHT の離散化誤差がほぼゼロになる。
    """
    f_hat = rng.standard_normal(L_MAX ** 2)
    return sht.backward(f_hat)  # shape: (N_THETA * N_PHI,)


def z_rotate_spatial(f: np.ndarray, k: int) -> np.ndarray:
    """
    球面信号を z 軸まわりに 2π·k/N_PHI だけ回転。

    等間隔 φ グリッドで k ステップのロールは補間誤差ゼロの厳密な回転になる。
    グリッド整合 (φ_0 が格子点に落ちる) が成立する前提。
    """
    return np.roll(f.reshape(N_THETA, N_PHI), k, axis=1).ravel()


# ─── T1: SHT 可逆性 ───────────────────────────────────────────────────────────

def test_sht_roundtrip():
    """
    スペクトル往復 f̂ → f → f̂ の誤差が許容値以内か確認。

    forward(backward(f̂)) = Y_w^T · Y · f̂ ≈ f̂
    球面調和基底の直交性が正しく実装されていれば Y_w^T·Y ≈ I になる。
    均等グリッドの求積誤差 (~6e-4 平均) が主要因。
    """
    rng = np.random.default_rng(0)
    sht = make_sht()

    errors = []
    for _ in range(10):
        f_hat = rng.standard_normal(L_MAX ** 2)
        f_hat_rec = sht.forward(sht.backward(f_hat))  # forward ∘ backward
        errors.append(np.abs(f_hat - f_hat_rec).mean())

    err = np.mean(errors)
    print(f"[T1] SHT round-trip mean error: {err:.2e}  (tol={TOL_SHT:.0e})")
    assert err < TOL_SHT, f"SHT round-trip error {err:.2e} > {TOL_SHT}"
    print("     ✓ PASSED")


# ─── T2: SphericalCNN の z軸回転不変性 ─────────────────────────────────────────

def test_spherical_cnn_z_rotation_invariance():
    """
    model(R·f) ≈ model(f)

    アーキテクチャが ℓ=0（全方向に不変なスカラー）のみを読み出す構造なので、
    出力は SO(3) 不変になるはず。
    z軸回転の k を変えて複数角度で検証する。
    """
    rng = np.random.default_rng(42)
    sht = make_sht()
    model = SphericalCNN(sht, channels=[1, 8, 16, 8], n_layers=3)
    model.eval()

    # バンド制限信号バッチ
    fs = np.stack([band_limited_signal(sht, rng) for _ in range(BATCH)])  # (B, N)
    x = torch.tensor(fs, dtype=torch.float32)

    with torch.no_grad():
        y = model(x)

    errors = []
    for k in [N_PHI // 8, N_PHI // 4, N_PHI // 2]:  # 45°, 90°, 180°
        fs_rot = np.stack([z_rotate_spatial(f, k) for f in fs])
        with torch.no_grad():
            y_rot = model(torch.tensor(fs_rot, dtype=torch.float32))
        err = (y - y_rot).abs().mean().item()
        errors.append(err)
        print(f"[T2]   k={k:2d} ({360*k//N_PHI:3d}°): invariance error = {err:.2e}")

    max_err = max(errors)
    print(f"[T2] max error: {max_err:.2e}  (tol={TOL_EQUIV:.0e})")
    assert max_err < TOL_EQUIV, f"Invariance error {max_err:.2e} > {TOL_EQUIV}"
    print("     ✓ PASSED")


# ─── T3: S2EquivariantLinear の z軸回転同変性 ───────────────────────────────────

def test_s2linear_z_rotation_equivariance():
    """
    "先に回転してから処理" = "処理してから回転" を空間域で検証。

    フロー:
      経路 A: f ──SHT──> f̂ ──layer──> ĝ ──iSHT──> g
                                          ↓ z-rotate(k)
                                        g_rolled

      経路 B: f ──z-rotate(k)──> Rf ──SHT──> R̂f ──layer──> ĝ' ──iSHT──> g'

    Schur の補題が言う「W_ℓ は m に非依存」から、layer は z 軸回転と可換になる。
    よって g_rolled ≈ g' のはず。
    """
    rng = np.random.default_rng(7)
    sht = make_sht()
    C_in, C_out = 4, 8
    layer = S2EquivariantLinear(L_MAX, C_in, C_out)
    layer.eval()

    # (B, C_in, N) のバンド制限信号を生成
    fs = np.array([[band_limited_signal(sht, rng) for _ in range(C_in)]
                   for _ in range(BATCH)])  # (B, C_in, N)

    def apply_layer_in_spatial(signal_bcn):
        """(B, C_in, N) → SHT → layer → iSHT → (B, C_out, N)"""
        # SHT: 全バッチ・全チャンネルに適用
        s_hat = np.array([[sht.forward(signal_bcn[b, c]) for c in range(C_in)]
                          for b in range(BATCH)])   # (B, C_in, L²)
        with torch.no_grad():
            g_hat = layer(torch.tensor(s_hat, dtype=torch.float32)).numpy()  # (B, C_out, L²)
        # iSHT
        return np.array([[sht.backward(g_hat[b, c]) for c in range(C_out)]
                         for b in range(BATCH)])    # (B, C_out, N)

    # 経路 A
    gs = apply_layer_in_spatial(fs)

    errors = []
    for k in [N_PHI // 8, N_PHI // 4, N_PHI // 3]:
        # 経路 B: 先に回転
        fs_rot = np.array([[z_rotate_spatial(fs[b, c], k) for c in range(C_in)]
                           for b in range(BATCH)])
        gs_rot = apply_layer_in_spatial(fs_rot)

        # 経路 A の出力を後から回転
        gs_rolled = np.array([[z_rotate_spatial(gs[b, c], k) for c in range(C_out)]
                              for b in range(BATCH)])

        err = np.abs(gs_rot - gs_rolled).mean()
        errors.append(err)
        print(f"[T3]   k={k:2d} ({360*k//N_PHI:3d}°): equivariance error = {err:.2e}")

    max_err = max(errors)
    print(f"[T3] max error: {max_err:.2e}  (tol={TOL_EQUIV:.0e})")
    assert max_err < TOL_EQUIV, f"Equivariance error {max_err:.2e} > {TOL_EQUIV}"
    print("     ✓ PASSED")


# ─── T4: ClebschGordanProduct の z軸回転同変性 ─────────────────────────────────

def spectral_z_rotate(f_hat: np.ndarray, phi0: float) -> np.ndarray:
    """
    z 軸回転 φ₀ に対応するスペクトル係数変換（実球面調和基底）。
    g(θ, φ) = f(θ, φ − φ₀) のスペクトル係数を返す。

    実球面調和の回転則（Wigner D 行列の 2×2 ブロック構造）:
      m = 0 : f̂(l,0) → f̂(l,0)              （不変）
      m > 0 : [f̂(l,m) ]   [cos mφ₀  −sin mφ₀] [f̂(l,m) ]
              [f̂(l,−m)] → [sin mφ₀   cos mφ₀] [f̂(l,−m)]

    f_hat の形状: (..., L_max²) — バッチ次元があっても動作する
    """
    f_out = f_hat.copy()
    for l in range(L_MAX):
        for m in range(1, l + 1):
            pos = l ** 2 + l + m   # (l, +m) のグローバルインデックス
            neg = l ** 2 + l - m   # (l, −m) のグローバルインデックス
            c, s = np.cos(m * phi0), np.sin(m * phi0)
            fp = f_hat[..., pos].copy()
            fn = f_hat[..., neg].copy()
            f_out[..., pos] = c * fp - s * fn
            f_out[..., neg] = s * fp + c * fn
    return f_out


def test_cg_product_z_rotation_equivariance():
    """
    CG(D(R)·f̂) ≈ D(R)·CG(f̂)  をスペクトル域で直接検証。

    D(R): 実球面調和基底での Wigner D 行列（z 軸回転は 2×2 ブロック対角）
    CG : ClebschGordanProduct 層

    Gaunt 係数の数値求積精度（~1e-4 程度）が誤差の主要因。
    空間域 ↔ スペクトル域の往復がないので SHT 離散化誤差は混入しない。
    """
    rng = np.random.default_rng(99)
    C_in, C_out = 2, 4
    layer = ClebschGordanProduct(L_MAX, C_in, C_out, exact_gaunt=True)
    layer.eval()

    # ランダムなバンド制限スペクトル係数 (B, C_in, L²)
    f_hat = rng.standard_normal((BATCH, C_in, L_MAX ** 2)).astype(np.float32)

    errors = []
    for k in [N_PHI // 8, N_PHI // 4, N_PHI // 3]:  # 45°, 90°, 120°
        phi0 = 2 * np.pi * k / N_PHI

        # 経路 A: 先に D(R) を適用してから CG 層
        f_rot = spectral_z_rotate(f_hat, phi0)             # (B, C_in, L²)
        with torch.no_grad():
            g_A = layer(torch.tensor(f_rot)).numpy()         # (B, C_out, L²)

        # 経路 B: 先に CG 層、後から D(R)
        with torch.no_grad():
            g = layer(torch.tensor(f_hat)).numpy()            # (B, C_out, L²)
        g_B = spectral_z_rotate(g, phi0)                    # (B, C_out, L²)

        err = np.abs(g_A - g_B).mean()
        errors.append(err)
        print(f"[T4]   k={k:2d} ({360*k//N_PHI:3d}°): CG equivariance error = {err:.2e}")

    max_err = max(errors)
    print(f"[T4] max error: {max_err:.2e}  (tol={TOL_EQUIV:.0e})")
    assert max_err < TOL_EQUIV, f"CG equivariance error {max_err:.2e} > {TOL_EQUIV}"
    print("     ✓ PASSED")


# ─── エントリポイント ─────────────────────────────────────────────────────────

# ─── SO(3) 任意回転ヘルパー ───────────────────────────────────────────────────

def spectral_rotate_so3(f_hat: np.ndarray,
                        alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    任意の SO(3) 回転 (ZYZ オイラー角) のスペクトル係数変換。

    g(n̂) = f(R^{-1}·n̂) の係数を返す（spectral_z_rotate と同じ規約）。

    変換: f̂_block @ D_ℓ(α,β,γ) を各 ℓ ブロックに適用。
    z 軸のみ (β=0) の場合は spectral_z_rotate(f̂, α) と一致する。

    f_hat: (..., L_MAX²)
    """
    f_out = f_hat.copy()
    for l in range(L_MAX):
        D = wigner_d_real(l, alpha, beta, gamma)  # (2l+1, 2l+1), float64
        s, e = l ** 2, l ** 2 + 2 * l + 1
        f_out[..., s:e] = f_hat[..., s:e] @ D     # D^T を右からかける形 = ĝ = D^T f̂
    return f_out


def _random_euler_angles(rng: np.random.Generator):
    """[0,2π) × [0,π] × [0,2π) から ZYZ オイラー角をサンプリング。"""
    alpha = rng.uniform(0, 2 * np.pi)
    beta  = np.arccos(rng.uniform(-1, 1))   # 一様分布 on S² → sin 重みを避ける
    gamma = rng.uniform(0, 2 * np.pi)
    return alpha, beta, gamma


# ─── T5: SphericalCNNv2 の z軸回転不変性 ───────────────────────────────────────

def test_spherical_cnn_v2_z_rotation_invariance():
    """
    SphericalCNNv2 (CG 層) の回転不変性を T2 と同条件で検証。

    v1 (NormNonlinearity) と v2 (ClebschGordanProduct) の不変性誤差を比較する。
    どちらも ℓ=0 のみ読み出すアーキテクチャなので理論的には完全不変のはず。
    """
    rng = np.random.default_rng(55)
    sht = make_sht()
    model_v2 = SphericalCNNv2(sht, channels=[1, 4, 4], n_layers=2, exact_gaunt=True)
    model_v2.eval()

    fs = np.stack([band_limited_signal(sht, rng) for _ in range(BATCH)])
    x  = torch.tensor(fs, dtype=torch.float32)

    with torch.no_grad():
        y = model_v2(x)

    errors = []
    for k in [N_PHI // 8, N_PHI // 4, N_PHI // 2]:  # 45°, 90°, 180°
        fs_rot = np.stack([z_rotate_spatial(f, k) for f in fs])
        with torch.no_grad():
            y_rot = model_v2(torch.tensor(fs_rot, dtype=torch.float32))
        err = (y - y_rot).abs().mean().item()
        errors.append(err)
        print(f"[T5]   k={k:2d} ({360*k//N_PHI:3d}°): invariance error = {err:.2e}")

    max_err = max(errors)
    print(f"[T5] max error: {max_err:.2e}  (tol={TOL_EQUIV:.0e})")
    assert max_err < TOL_EQUIV, f"v2 invariance error {max_err:.2e} > {TOL_EQUIV}"
    print("     ✓ PASSED")


# ─── エントリポイント ─────────────────────────────────────────────────────────

# ─── T6: 任意 SO(3) 回転での同変性・不変性 ────────────────────────────────────

N_ROTATIONS = 5   # ランダム回転の試行回数

def test_so3_equivariance():
    """
    任意の SO(3) 回転（ZYZ Euler 角をランダムサンプリング）に対して
    S2EquivariantLinear と ClebschGordanProduct の同変性を検証する。

    テスト構造（各回転 R ごと）:
      S2Linear:  layer(D(R)·f̂) ≈ D(R)·layer(f̂)
      CGProduct: CG(D(R)·f̂)   ≈ D(R)·CG(f̂)

    D(R) はスペクトル域の各 ℓ ブロックに Wigner D 行列を適用する演算子。
    z 軸テスト (T3/T4) は float32 精度の限界 ~1e-7 だったが、
    任意回転では Wigner D 行列の float64 計算と float32 層の精度の積で
    同程度 ~1e-5 の誤差を想定する。
    """
    rng = np.random.default_rng(2025)
    sht = make_sht()
    C_in, C_out = 2, 4

    s2_layer = S2EquivariantLinear(L_MAX, C_in, C_out)
    cg_layer = ClebschGordanProduct(L_MAX, C_in, C_in, exact_gaunt=True)  # self-product
    s2_layer.eval(); cg_layer.eval()

    # バンド制限スペクトル係数 (B, C_in, L²)
    f_hat = rng.standard_normal((BATCH, C_in, L_MAX ** 2)).astype(np.float32)

    s2_errors, cg_errors = [], []

    for trial in range(N_ROTATIONS):
        alpha, beta, gamma = _random_euler_angles(rng)
        label = f"α={np.degrees(alpha):.1f}° β={np.degrees(beta):.1f}° γ={np.degrees(gamma):.1f}°"

        # 回転を適用した f̂
        f_rot = spectral_rotate_so3(f_hat, alpha, beta, gamma)  # float64 (broadcast OK)
        f_rot32 = f_rot.astype(np.float32)

        # ── S2EquivariantLinear ──────────────────────────────────────────────
        with torch.no_grad():
            # 経路 A: 先に回転、後で層
            gA_s2 = s2_layer(torch.tensor(f_rot32)).numpy()          # (B, C_out, L²)
            # 経路 B: 先に層、後で回転
            gB_s2 = spectral_rotate_so3(
                s2_layer(torch.tensor(f_hat)).numpy(), alpha, beta, gamma)

        err_s2 = np.abs(gA_s2 - gB_s2).mean()
        s2_errors.append(err_s2)

        # ── ClebschGordanProduct ─────────────────────────────────────────────
        with torch.no_grad():
            gA_cg = cg_layer(torch.tensor(f_rot32)).numpy()
            gB_cg = spectral_rotate_so3(
                cg_layer(torch.tensor(f_hat)).numpy(), alpha, beta, gamma)

        err_cg = np.abs(gA_cg - gB_cg).mean()
        cg_errors.append(err_cg)

        print(f"[T6] {label}")
        print(f"     S2Linear err={err_s2:.2e}  CG err={err_cg:.2e}")

    max_s2 = max(s2_errors)
    max_cg = max(cg_errors)
    print(f"[T6] max  S2Linear: {max_s2:.2e}  (tol={TOL_EQUIV:.0e})")
    print(f"[T6] max  CGProduct: {max_cg:.2e}  (tol={TOL_EQUIV:.0e})")
    assert max_s2 < TOL_EQUIV, f"S2Linear SO3 equivariance {max_s2:.2e} > {TOL_EQUIV}"
    assert max_cg < TOL_EQUIV, f"CGProduct SO3 equivariance {max_cg:.2e} > {TOL_EQUIV}"
    print("     ✓ PASSED")


# ─── エントリポイント ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("球面CNN 同変性テスト")
    print(f"  L_max={L_MAX}, grid=({N_THETA}×{N_PHI}), batch={BATCH}")
    print("=" * 55)
    test_sht_roundtrip()
    print()
    test_spherical_cnn_z_rotation_invariance()
    print()
    test_s2linear_z_rotation_equivariance()
    print()
    test_cg_product_z_rotation_equivariance()
    print()
    test_spherical_cnn_v2_z_rotation_invariance()
    print()
    test_so3_equivariance()
    print()
    print("=" * 55)
    print("All tests passed ✓")
    print("=" * 55)
