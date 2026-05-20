"""
ゲージ同変性のユニットテスト

検証内容:
  T1. GaugeConv: 単一頂点のゲージ回転 φ に対して
      f'_v[n] → e^{-inφ} · f'_v[n] が成立するか
  T2. GaugeConv: ゲージ回転した頂点以外は出力が変化しないか
  T3. GaugeConvTypeN: 型 n の複素入力に対して同じゲージ同変性が成立するか
  T4. IcosGaugeCNN: 任意のゲージ変換に対して出力が不変か（ゲージ不変性）
"""

import numpy as np
import torch
import torch.nn as nn
from gauge_cnn import (build_gauge_data, GaugeConv, GaugeConvTypeN,
                       GaugeNorm, InvariantPool, IcosGaugeCNN)


ATOL = 1e-5


def _rotate_gauge_at(nb_ang: torch.Tensor, v0: int, phi: float) -> torch.Tensor:
    """
    頂点 v0 のゲージを φ だけ回転した接続角テンソルを返す。

    ゲージ変換 φ_v0 は v0 の局所フレームを φ だけ回転させる。
    v0 の「出辺」の接続角 α_{v0→w} → α_{v0→w} - φ（v0 の基準が変わる）。
    他の頂点の角度は変化しない。
    """
    ang = nb_ang.clone()
    ang[v0, :] -= phi
    return ang


# ────────────────────────────────────────────────────────────────
# T1 & T2: GaugeConv の同変性
# ────────────────────────────────────────────────────────────────

def test_gauge_conv_equivariance():
    """
    T1: v0 のゲージ回転 φ → f'[v0, n] = e^{-inφ} · f[v0, n]
    T2: v ≠ v0 の出力は変化しない
    """
    print("=== T1 & T2: GaugeConv ゲージ同変性 ===")
    device = torch.device('cpu')
    nb_idx, nb_ang, _, _ = build_gauge_data(device)

    C_IN, C_OUT, N_TYPES = 4, 8, 5
    B, V = 3, 12
    v0, phi = 3, 1.23

    conv = GaugeConv(C_IN, C_OUT, N_TYPES, nb_idx, nb_ang)
    conv.eval()
    x = torch.randn(B, V, C_IN)

    with torch.no_grad():
        out_orig = conv(x)  # (B,V,N,C,2)

    # 回転後の接続角で同一の重みを使って計算
    nb_ang_rot = _rotate_gauge_at(nb_ang, v0, phi)
    conv_rot = GaugeConv(C_IN, C_OUT, N_TYPES, nb_idx, nb_ang_rot)
    conv_rot.weight_real.data = conv.weight_real.data.clone()
    conv_rot.weight_imag.data = conv.weight_imag.data.clone()
    conv_rot.eval()

    with torch.no_grad():
        out_rot = conv_rot(x)  # (B,V,N,C,2)

    # T1: v0 での変換チェック f'[v0,n] = e^{-inφ} · f[v0,n]
    all_pass = True
    for n in range(N_TYPES):
        c, s = np.cos(-n * phi), np.sin(-n * phi)
        orig_r = out_orig[:, v0, n, :, 0]
        orig_i = out_orig[:, v0, n, :, 1]
        exp_r  = c * orig_r - s * orig_i   # e^{-inφ} の実部
        exp_i  = c * orig_i + s * orig_r   # e^{-inφ} の虚部
        err = max(
            (out_rot[:, v0, n, :, 0] - exp_r).abs().max().item(),
            (out_rot[:, v0, n, :, 1] - exp_i).abs().max().item(),
        )
        ok = err < ATOL
        all_pass = all_pass and ok
        print(f"  T1 n={n}: e^{{-i{n}φ}} 変換誤差 = {err:.2e}  {'✓' if ok else '✗'}")

    # T2: v0 以外の頂点が不変か
    other_verts = [v for v in range(V) if v != v0]
    err_others = (out_rot[:, other_verts, :, :, :] - out_orig[:, other_verts, :, :, :]).abs().max().item()
    ok2 = err_others < ATOL
    all_pass = all_pass and ok2
    print(f"  T2 v≠v0 不変誤差 = {err_others:.2e}  {'✓' if ok2 else '✗'}")

    return all_pass


# ────────────────────────────────────────────────────────────────
# T3: GaugeConvTypeN の同変性
# ────────────────────────────────────────────────────────────────

def test_gauge_conv_type_n_equivariance():
    """
    T3: 複素 type-n 入力に対して GaugeConvTypeN が同じゲージ同変性を持つか
    """
    print("\n=== T3: GaugeConvTypeN ゲージ同変性 ===")
    device = torch.device('cpu')
    nb_idx, nb_ang, _, _ = build_gauge_data(device)

    C_IN, C_OUT, N_TYPES = 6, 6, 4
    B, V = 2, 12
    v0, phi = 7, -0.88

    conv_n = GaugeConvTypeN(C_IN, C_OUT, N_TYPES, nb_idx, nb_ang)
    conv_n.eval()

    # ランダムな複素入力（type-n 特徴量をシミュレート）
    x = torch.randn(B, V, N_TYPES, C_IN, 2)

    with torch.no_grad():
        out_orig = conv_n(x)  # (B,V,N,C_OUT,2)

    nb_ang_rot = _rotate_gauge_at(nb_ang, v0, phi)
    conv_n_rot = GaugeConvTypeN(C_IN, C_OUT, N_TYPES, nb_idx, nb_ang_rot)
    conv_n_rot.weight_real.data = conv_n.weight_real.data.clone()
    conv_n_rot.weight_imag.data = conv_n.weight_imag.data.clone()
    conv_n_rot.eval()

    with torch.no_grad():
        out_rot = conv_n_rot(x)

    all_pass = True
    for n in range(N_TYPES):
        c, s = np.cos(-n * phi), np.sin(-n * phi)
        orig_r = out_orig[:, v0, n, :, 0]
        orig_i = out_orig[:, v0, n, :, 1]
        exp_r  = c * orig_r - s * orig_i
        exp_i  = c * orig_i + s * orig_r
        err = max(
            (out_rot[:, v0, n, :, 0] - exp_r).abs().max().item(),
            (out_rot[:, v0, n, :, 1] - exp_i).abs().max().item(),
        )
        ok = err < ATOL
        all_pass = all_pass and ok
        print(f"  T3 n={n}: 誤差 = {err:.2e}  {'✓' if ok else '✗'}")

    return all_pass


# ────────────────────────────────────────────────────────────────
# T4: IcosGaugeCNN のゲージ不変性
# ────────────────────────────────────────────────────────────────

def test_icos_gauge_cnn_invariance():
    """
    T4: 全頂点に異なるゲージ変換を適用してもモデル出力が変わらないか。

    ゲージ同変層 + InvariantPool の組み合わせで出力はゲージ不変になる。
    ランダムな φ_v を全頂点に適用して出力の差を確認する。
    """
    print("\n=== T4: IcosGaugeCNN ゲージ不変性 ===")
    device = torch.device('cpu')
    nb_idx, nb_ang, _, _ = build_gauge_data(device)

    C_IN, C_HID, N_TYPES = 3, 8, 4
    B, V = 4, 12

    model = IcosGaugeCNN(C_IN, C_HID, N_TYPES, n_out=2,
                         neighbor_idx=nb_idx, neighbor_angles=nb_ang)
    model.eval()
    x = torch.randn(B, V, C_IN)

    with torch.no_grad():
        out_orig = model(x)  # (B,V,2)

    # 全頂点にランダムなゲージ変換を適用
    phis = torch.rand(V) * 2 * np.pi  # (V,) ランダム角度

    nb_ang_rot = nb_ang.clone()
    for v in range(V):
        nb_ang_rot[v, :] -= phis[v].item()   # α_{v→w} → α_{v→w} - φ_v

    model_rot = IcosGaugeCNN(C_IN, C_HID, N_TYPES, n_out=2,
                              neighbor_idx=nb_idx, neighbor_angles=nb_ang_rot)
    # 同一パラメータをコピー
    model_rot.load_state_dict(model.state_dict())
    model_rot.eval()

    with torch.no_grad():
        out_rot = model_rot(x)

    err = (out_rot - out_orig).abs().max().item()
    ok = err < ATOL
    print(f"  T4 全頂点ゲージ回転後の出力誤差 = {err:.2e}  {'✓' if ok else '✗'}")
    if not ok:
        print(f"  !! out_orig[:2,:3]: {out_orig[:2,:3].detach().numpy().round(4)}")
        print(f"  !! out_rot  [:2,:3]: {out_rot[:2,:3].detach().numpy().round(4)}")

    return ok


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []
    results.append(test_gauge_conv_equivariance())
    results.append(test_gauge_conv_type_n_equivariance())
    results.append(test_icos_gauge_cnn_invariance())

    print()
    if all(results):
        print("全テスト通過 ✓")
    else:
        n_fail = results.count(False)
        print(f"{n_fail} テスト失敗 ✗")
        raise SystemExit(1)
