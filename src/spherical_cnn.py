"""
球面CNN — 自前実装

数学的構造:
  Peter-Weyl定理  →  Y行列（球面調和基底行列）を構築
  畳み込み定理    →  SHT空間で ℓ ブロックごとの積
  Schurの補題     →  同変フィルタ = W_ℓ（m非依存の行列）
"""

from math import factorial as _fact, sqrt as _sqrt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import sph_harm_y


# ============================================================
# 球面調和基底行列
# ============================================================

def build_Y_matrix(L_max: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Y[n, (ℓ,m)]  shape: (N, L_max²)  — 実球面調和関数

    Peter-Weylの基底をグリッド点で評価した行列。
    列のインデックス順: ℓ=0(m=0), ℓ=1(m=-1,0,1), ℓ=2(m=-2,...,2), ...

    複素球面調和関数 Y_l^m から実基底を構成:
      m > 0:  Re(Y_l^m) · √2
      m = 0:  Y_l^0   （もともと実数）
      m < 0:  Im(Y_l^{|m|}) · √2

    これがSHT・逆SHT・同変層すべての土台になる。
    """
    N = len(theta)
    n_basis = L_max ** 2  # Σ_{ℓ=0}^{L-1}(2ℓ+1) = L²

    Y = np.zeros((N, n_basis))
    idx = 0
    for l in range(L_max):
        for m in range(-l, l + 1):
            Y_complex = sph_harm_y(l, abs(m), theta, phi)
            if m > 0:
                Y[:, idx] = Y_complex.real * np.sqrt(2)
            elif m == 0:
                Y[:, idx] = Y_complex.real
            else:  # m < 0
                Y[:, idx] = Y_complex.imag * np.sqrt(2)
            idx += 1
    return Y


# ============================================================
# 球面調和変換 (SHT)
# ============================================================

class SphericalHarmonicTransform:
    """
    Forward (解析):  f(θ,φ) [N]  →  f̂(ℓ,m) [L²]
    Backward (合成): f̂(ℓ,m) [L²] →  f(θ,φ) [N]

    Forward の数式:
      f̂ = Y† · diag(w) · f
      w_n = sin(θ_n) · Δθ · Δφ  ← dΩ = sinθ dθ dφ の離散化

    Backward の数式:
      f = Y · f̂  ← Peter-Weyl逆展開
    """

    def __init__(self, L_max: int, n_theta: int, n_phi: int):
        self.L_max = L_max
        self.grid_shape = (n_theta, n_phi)

        # 等間隔グリッド（半グリッドシフトで端点を避ける）
        theta = np.linspace(0, np.pi,   n_theta, endpoint=False) + np.pi / (2 * n_theta)
        phi   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)

        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        theta_flat = Theta.ravel()
        phi_flat   = Phi.ravel()

        # 求積重み: w_n = sin(θ_n) · Δθ · Δφ
        dtheta = np.pi / n_theta
        dphi   = 2 * np.pi / n_phi
        w = np.sin(theta_flat) * dtheta * dphi  # shape: (N,)

        # Y行列 (Peter-Weylの核心) — 実数値
        Y = build_Y_matrix(L_max, theta_flat, phi_flat)  # (N, L²)

        # 解析行列: Y^T · diag(w)  →  shape (L², N)
        self.Y_analysis  = Y.T * w[np.newaxis, :]  # 実基底なので転置=随伴
        self.Y_synthesis = Y  # shape (N, L²)

    def forward(self, f: np.ndarray) -> np.ndarray:
        """f: (N,) → f̂: (L²,)  実数値"""
        return self.Y_analysis @ f

    def backward(self, f_hat: np.ndarray) -> np.ndarray:
        """f̂: (L²,) → f: (N,)  実数値"""
        return self.Y_synthesis @ f_hat


# ============================================================
# 同変線形層 — Schurの補題の直接実装
# ============================================================

class S2EquivariantLinear(nn.Module):
    """
    Schurの補題が言うこと:
      SO(3)同変な線形写像 T は、各既約表現 ρ^(ℓ) 上でスカラー倍のみ許される。
      多チャンネルに拡張すると: T|_{V^(ℓ)} = W_ℓ ⊗ I_{2ℓ+1}

    実装上のポイント:
      f̂^out(ℓ, m) = W_ℓ · f̂^in(ℓ, m)   ← W_ℓ は m に依存しない
      einsum('oi,bim->bom', W_ℓ, f̂[:,:,ℓブロック])

    パラメータ数の比較（L_max=5, C=8 の場合）:
      Schur制約後:   L_max × C² =   5 × 64 =   320
      制約なし:      Σ(2ℓ+1)² × C² = 55 × 64 = 3520  （11倍）
    """

    def __init__(self, L_max: int, C_in: int, C_out: int):
        super().__init__()
        self.L_max = L_max

        # W_ℓ ∈ R^{C_out × C_in}  を ℓ ごとに独立に学習
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(C_out, C_in) / C_in ** 0.5)
            for _ in range(L_max)
        ])

    def forward(self, f_hat: torch.Tensor) -> torch.Tensor:
        """
        f_hat: (B, C_in, L²)
        return: (B, C_out, L²)

        ℓブロックごとに W_ℓ を適用。ブロック間は独立（Schurの補題）。
        """
        B, C_in, _ = f_hat.shape
        C_out = self.W[0].shape[0]
        out = torch.zeros(B, C_out, self.L_max ** 2,
                          dtype=f_hat.dtype, device=f_hat.device)
        idx = 0
        for l, W_l in enumerate(self.W):
            m_count = 2 * l + 1
            # 同じ W_l を m=-l,...,+l の全スロットに一律適用
            # ↑ ここが「mに依存しない」= Schurの補題の実装箇所
            out[:, :, idx:idx + m_count] = torch.einsum(
                'oi,bim->bom', W_l, f_hat[:, :, idx:idx + m_count]
            )
            idx += m_count
        return out


# ============================================================
# Normノンリニアリティ — 同変性を保つ活性化
# ============================================================

class NormNonlinearity(nn.Module):
    """
    各 ℓ ブロックのノルムにだけ非線形を適用することで同変性を保つ。

    数式:  f̂^out(ℓ,m) = σ(‖f̂(ℓ)‖) / ‖f̂(ℓ)‖ · f̂^in(ℓ,m)

    方向（m成分の比率）は変えず、大きさだけを変換する。
    SO(3)同変性はこの操作でも保たれる。
    """

    def __init__(self, L_max: int):
        super().__init__()
        self.L_max = L_max
        self.activation = nn.functional.relu

    def forward(self, f_hat: torch.Tensor) -> torch.Tensor:
        """f_hat: (B, C, L²)"""
        out = torch.zeros_like(f_hat)
        idx = 0
        for l in range(self.L_max):
            m_count = 2 * l + 1
            block = f_hat[:, :, idx:idx + m_count]          # (B, C, 2ℓ+1)
            norm  = block.norm(dim=-1, keepdim=True) + 1e-8  # (B, C, 1)
            scale = self.activation(norm) / norm              # 非線形はノルムだけに
            out[:, :, idx:idx + m_count] = scale * block
            idx += m_count
        return out


# ============================================================
# Gaunt 係数 & Clebsch-Gordan テンソル積層
# ============================================================

def _real_sph_harm(l: int, m: int,
                   theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """実球面調和関数 Y^R_{l,m}(θ,φ)"""
    Yc = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        return Yc.real * np.sqrt(2)
    elif m == 0:
        return Yc.real
    else:
        return Yc.imag * np.sqrt(2)


def compute_gaunt_matrix(l1: int, l2: int, l: int,
                         fine_n: int = 64) -> np.ndarray:
    """
    Gaunt 係数行列を数値求積で計算（精度 ~1e-4 at L_max=6）。

    G[m1_idx, m2_idx, m_idx] = ∫ Y^R_{l1,m1} Y^R_{l2,m2} Y^R_{l,m} dΩ

    Returns: shape (2*l1+1, 2*l2+1, 2*l+1), dtype float64
    """
    theta = (np.linspace(0, np.pi, fine_n, endpoint=False)
             + np.pi / (2 * fine_n))
    phi   = np.linspace(0, 2 * np.pi, 2 * fine_n, endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    t_flat, p_flat = Theta.ravel(), Phi.ravel()
    w = np.sin(t_flat) * (np.pi / fine_n) * (np.pi / fine_n)  # dθ·dφ

    Y1 = np.stack([_real_sph_harm(l1, m, t_flat, p_flat)
                   for m in range(-l1, l1 + 1)], axis=1)   # (N, 2l1+1)
    Y2 = np.stack([_real_sph_harm(l2, m, t_flat, p_flat)
                   for m in range(-l2, l2 + 1)], axis=1)   # (N, 2l2+1)
    Yl = np.stack([_real_sph_harm(l,  m, t_flat, p_flat)
                   for m in range(-l,  l  + 1)], axis=1)   # (N, 2l+1)

    return np.einsum('ni,nj,nk,n->ijk', Y1, Y2, Yl, w, optimize=True)


def compute_gaunt_matrix_exact(l1: int, l2: int, l: int) -> np.ndarray:
    """
    Gaunt 係数行列を sympy の Wigner 3-j 記号で解析的に計算（精度 ~1e-14）。

    複素球面調和の三重積分:
      G^C[m1,m2,m] = ∫ Y^C_{l1,m1} Y^C_{l2,m2} Y^C_{l,m} dΩ
                   = sqrt((2l1+1)(2l2+1)(2l+1)/(4π)) * W3j(l1,l2,l;0,0,0)
                     * W3j(l1,l2,l;m1,m2,m)  （m1+m2+m=0 でなければゼロ）

    実基底への変換:
      G^R[m1,m2,m] = Σ_{a,b,c} U1[m1,a] U2[m2,b] U[m,c] * G^C[a,b,c]
      （U は _cob_complex_to_real で定義される複素→実基底変換行列）

    Returns: shape (2*l1+1, 2*l2+1, 2*l+1), dtype float64
    """
    from sympy.physics.wigner import wigner_3j
    from sympy import sqrt as sympy_sqrt, pi as sympy_pi, N as sympy_N

    n1, n2, n = 2*l1+1, 2*l2+1, 2*l+1

    # プリファクター sqrt((2l1+1)(2l2+1)(2l+1)/(4π)) * W3j(l1,l2,l;0,0,0)
    pf = float(sympy_N(
        sympy_sqrt((2*l1+1) * (2*l2+1) * (2*l+1))
        / sympy_sqrt(4 * sympy_pi)
        * wigner_3j(l1, l2, l, 0, 0, 0)
    ))

    # 複素 Gaunt 行列（選択則: m1+m2+m=0）
    G_complex = np.zeros((n1, n2, n), dtype=complex)
    for i1, m1 in enumerate(range(-l1, l1+1)):
        for i2, m2 in enumerate(range(-l2, l2+1)):
            m = -(m1 + m2)
            if abs(m) > l:
                continue
            w3j = float(sympy_N(wigner_3j(l1, l2, l, m1, m2, m)))
            G_complex[i1, i2, m + l] = pf * w3j

    # 複素 → 実球面調和基底への変換
    U1 = _cob_complex_to_real(l1)
    U2 = _cob_complex_to_real(l2)
    U  = _cob_complex_to_real(l)

    G_real = np.einsum('ia,jb,kc,abc->ijk', U1, U2, U, G_complex).real
    return G_real.astype(np.float64)


class ClebschGordanProduct(nn.Module):
    """
    Clebsch-Gordan テンソル積層

    入力 f̂ (B, C_in, L²) を自己結合して f̂_out (B, C_out, L²) を生成。

    各出力チャンネル (l_out, m) の値:
      f̂_out[b, c_out, l_out, m]
        = Σ_{(l1,l2)} Σ_{c1,c2} W[l_out,l1,l2][c_out, c1, c2]
          · Σ_{m1,m2} G[l1,m1; l2,m2; l_out,m] · f̂[b,c1,l1,m1] · f̂[b,c2,l2,m2]

    G: Gaunt 係数（precomputed・固定）
    W: 学習可能スカラー（パスごとに (C_out, C_in, C_in) の行列）

    NormNonlinearity との違い:
      NormNonlinearity: ℓ ブロック内のノルムだけ変換（ℓ 間の混合なし）
      ClebschGordanProduct: 異なる ℓ 同士を結合して新しい ℓ を生成
        例) ℓ=1 ⊗ ℓ=1 = ℓ=0 ⊕ ℓ=1 ⊕ ℓ=2

    SO(3) 同変性の根拠:
      球面調和関数の積則 Y_{l1m1}·Y_{l2m2} = Σ_{lm} G[l1m1,l2m2,lm] Y_{lm}
      から、G が SO(3) の表現行列と可換であることが保証される。
    """

    def __init__(self, L_max: int, C_in: int, C_out: int,
                 fine_n: int = 64, exact_gaunt: bool = True):
        super().__init__()
        self.L_max = L_max
        self.C_out = C_out

        # 有効パス (l_out, l1, l2) を列挙（三角不等式を満たすもの）
        self.paths = [
            (lo, l1, l2)
            for lo in range(L_max)
            for l1 in range(L_max)
            for l2 in range(L_max)
            if abs(l1 - l2) <= lo <= l1 + l2
        ]
        n_paths = len(self.paths)

        # パスごとに Gaunt 行列（buffer）と学習可能重みを登録
        # exact_gaunt=True: sympy Wigner 3-j で解析的計算 (~1e-14)
        # exact_gaunt=False: 数値求積 fine_n で計算 (~1e-4)
        weights: dict[str, nn.Parameter] = {}
        for lo, l1, l2 in self.paths:
            G = (compute_gaunt_matrix_exact(l1, l2, lo) if exact_gaunt
                 else compute_gaunt_matrix(l1, l2, lo, fine_n))
            self.register_buffer(f'G_{lo}_{l1}_{l2}',
                                 torch.tensor(G, dtype=torch.float32))
            weights[f'{lo}_{l1}_{l2}'] = nn.Parameter(
                torch.randn(C_out, C_in, C_in) / np.sqrt(C_in * n_paths)
            )
        self.weights = nn.ParameterDict(weights)

    def forward(self, f_hat: torch.Tensor) -> torch.Tensor:
        """
        f_hat: (B, C_in, L²)
        return: (B, C_out, L²)

        各パス (l_out, l1, l2) で:
          1. f̂₁ ⊗ f̂₂ を Gaunt 行列で収縮 → (B, C_in, C_in, 2l_out+1)
          2. W で チャンネル混合        → (B, C_out, 2l_out+1)
          3. l_out ブロックに加算
        """
        B = f_hat.size(0)
        out = torch.zeros(B, self.C_out, self.L_max ** 2,
                          dtype=f_hat.dtype, device=f_hat.device)

        for lo, l1, l2 in self.paths:
            G = getattr(self, f'G_{lo}_{l1}_{l2}')   # (2l1+1, 2l2+1, 2lo+1)
            W = self.weights[f'{lo}_{l1}_{l2}']        # (C_out, C_in, C_in)

            b1 = f_hat[:, :, l1**2 : l1**2 + 2*l1 + 1]  # (B, C_in, 2l1+1)
            b2 = f_hat[:, :, l2**2 : l2**2 + 2*l2 + 1]  # (B, C_in, 2l2+1)

            # Gaunt 収縮: (B, C_in, C_in, 2lo+1)
            coupled = torch.einsum('bci,bdj,ijk->bcdk', b1, b2, G)
            # チャンネル混合: (B, C_out, 2lo+1)
            out_block = torch.einsum('ocd,bcdk->bok', W, coupled)

            s = lo ** 2
            out[:, :, s : s + 2*lo + 1] = out[:, :, s : s + 2*lo + 1] + out_block

        return out


# ============================================================
# 球面CNN
# ============================================================

class SphericalCNN(nn.Module):
    """
    アーキテクチャ:

      f(θ,φ) [空間域]
        ↓ SHT（Peter-Weyl展開）
      f̂(ℓ,m) [スペクトル域]
        ↓ [S2EquivariantLinear → NormNonlinearity] × n_layers
      f̂^out(ℓ,m)
        ↓ ℓ=0成分だけ取り出す（不変スカラー）または 逆SHT
      出力
    """

    def __init__(self, sht: SphericalHarmonicTransform,
                 channels: list[int], n_layers: int = 2):
        super().__init__()
        L_max = sht.L_max
        self.sht = sht

        layers = []
        for i in range(n_layers):
            layers.append(S2EquivariantLinear(L_max, channels[i], channels[i + 1]))
            layers.append(NormNonlinearity(L_max))
        self.layers = nn.Sequential(*layers)

        # 最終出力: ℓ=0 チャンネル（全方向に不変なスカラー特徴）
        self.readout = nn.Linear(channels[-1], 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B, N)  ← N = n_theta × n_phi グリッド点
        return: (B, 1)
        """
        B, N = f.shape

        # SHT: 各バッチ・各チャンネルに適用（実数のみ）
        Y_a = torch.tensor(self.sht.Y_analysis, dtype=torch.float32, device=f.device)
        f_hat = torch.einsum('kn,bn->bk', Y_a, f)  # (B, L²)
        f_hat = f_hat.unsqueeze(1)                        # (B, 1, L²) ← チャンネル次元

        # 同変処理
        f_hat = self.layers(f_hat)                        # (B, C, L²)

        # ℓ=0（m=0）はSO(3)不変スカラー ← idx=0
        invariant = f_hat[:, :, 0]                        # (B, C)
        return self.readout(invariant)                     # (B, 1)


# ============================================================
# Wigner D 行列 — 任意の SO(3) 回転のスペクトル域表現
# ============================================================

def _cob_complex_to_real(l: int) -> np.ndarray:
    """
    複素 → 実球面調和関数への変換行列 U (size 2l+1)。
    Y^R_{l,m} = Σ_{m'} U_{m,m'} Y^C_{l,m'}

    実装している Y^R 規約:
      m > 0: √2 Re(Y^C_{l,m})  = (Y^C_{l,m} + (−1)^m Y^C_{l,−m}) / √2
      m = 0: Y^C_{l,0}
      m < 0: √2 Im(Y^C_{l,|m|}) = (−i Y^C_{l,|m|} + i(−1)^|m| Y^C_{l,−|m|}) / √2
    """
    n, rt2 = 2 * l + 1, 1.0 / _sqrt(2)
    U = np.zeros((n, n), dtype=complex)
    for row, m in enumerate(range(-l, l + 1)):
        if m == 0:
            U[row, l] = 1.0
        elif m > 0:
            U[row, l + m] = rt2
            U[row, l - m] = (-1) ** m * rt2
        else:
            ab = abs(m)
            U[row, l + ab] = -1j * rt2
            U[row, l - ab] =  1j * (-1) ** ab * rt2
    return U


def wigner_small_d(l: int, beta: float) -> np.ndarray:
    """
    小 Wigner d 行列 d^l_{m'm}(β) — shape (2l+1, 2l+1), 実数値。
    行インデックス = m'+l (m'=−l..+l), 列インデックス = m+l。

    Condon-Shortley 符号規約:
      d^l_{m'm}(β) = √[(l+m')!(l−m')!(l+m)!(l−m)!]
                     × Σ_s (−1)^{m'−m+s} / [s!(l+m−s)!(m'−m+s)!(l−m'−s)!]
                     × cos^{2l+m−m'−2s}(β/2) × sin^{m'−m+2s}(β/2)
    """
    n = 2 * l + 1
    d = np.zeros((n, n))
    cb, sb = np.cos(beta / 2), np.sin(beta / 2)
    for mp_idx, mp in enumerate(range(-l, l + 1)):
        for m_idx, m in enumerate(range(-l, l + 1)):
            pf = _sqrt(_fact(l+mp) * _fact(l-mp) * _fact(l+m) * _fact(l-m))
            s_min, s_max = max(0, m - mp), min(l + m, l - mp)
            val = 0.0
            for s in range(s_min, s_max + 1):
                sign = (-1) ** (mp - m + s)
                denom = (_fact(l+m-s) * _fact(s)
                         * _fact(mp-m+s) * _fact(l-mp-s))
                cp = 2 * l + m - mp - 2 * s
                sp = mp - m + 2 * s
                val += sign / denom * cb ** cp * sb ** sp
            d[mp_idx, m_idx] = pf * val
    return d


def wigner_d_real(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    実球面調和基底の Wigner D 行列 D^{(l,real)}(α,β,γ) — shape (2l+1, 2l+1)。
    ZYZ オイラー角: R = R_z(α) R_y(β) R_z(γ)

    変換則:
      f(Rn̂) のスペクトル係数 ĝ = f̂ @ D   （f̂ は行ベクトル）
      または ĝ_ℓ = f̂_ℓ @ D_ℓ  を各 ℓ ブロックに適用

    z 軸回転 R_z(φ) では D_ℓ の (m, −m) × (m, −m) 部分行列が
      [[cos mφ, sin mφ], [−sin mφ, cos mφ]]
    になり、spectral_z_rotate と同じ規約になる。

    Returns: 実直交行列 (D @ D.T = I)
    """
    d = wigner_small_d(l, beta)
    ms = np.arange(-l, l + 1, dtype=float)
    D_complex = np.exp(-1j * ms * alpha)[:, None] * d * np.exp(-1j * ms * gamma)[None, :]
    U = _cob_complex_to_real(l)
    D_real = U @ D_complex @ U.conj().T
    return D_real.real   # 虚部は ~1e-15 の数値誤差のみ


# ============================================================
# SphericalCNNv2 — CG 層を非線形性として使う拡張版
# ============================================================

class SphericalCNNv2(nn.Module):
    """
    SphericalCNN v2: NormNonlinearity → ClebschGordanProduct

    v1 との違い:
      v1: [S2Linear → NormNonlinearity] × n_layers
            NormNonlinearity はℓブロック内のノルムだけを変換。ℓ間の混合なし。
      v2: [S2Linear → ClebschGordanProduct] × n_layers
            CG 層は f̂ の自己テンソル積を計算し、異なるℓ値を結合できる。
            例: ℓ=1 ⊗ ℓ=1 → ℓ=0（内積相当）, ℓ=1（外積相当）, ℓ=2（テンソル）

    どちらも ℓ=0 成分だけ読み出すので SO(3) 不変スカラーを出力する。
    """

    def __init__(self, sht: SphericalHarmonicTransform,
                 channels: list[int], n_layers: int = 2,
                 exact_gaunt: bool = True):
        super().__init__()
        L_max = sht.L_max
        self.sht = sht

        layers = []
        for i in range(n_layers):
            layers.append(S2EquivariantLinear(L_max, channels[i], channels[i + 1]))
            layers.append(ClebschGordanProduct(L_max, channels[i + 1], channels[i + 1],
                                               exact_gaunt=exact_gaunt))
        self.layers = nn.Sequential(*layers)

        self.readout = nn.Linear(channels[-1], 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f: (B, N) → (B, 1)"""
        Y_a = torch.tensor(self.sht.Y_analysis, dtype=torch.float32, device=f.device)
        f_hat = torch.einsum('kn,bn->bk', Y_a, f).unsqueeze(1)  # (B, 1, L²)
        f_hat = self.layers(f_hat)                                # (B, C, L²)
        return self.readout(f_hat[:, :, 0])                       # ℓ=0 → (B, 1)


# ============================================================
# 動作確認
# ============================================================

if __name__ == '__main__':
    L_max   = 6
    n_theta = 32
    n_phi   = 64
    B       = 4
    C       = [1, 4, 4]   # 小さいチャンネルで動作確認

    sht = SphericalHarmonicTransform(L_max, n_theta, n_phi)

    # バンド制限入力 (B, N)
    x = torch.tensor(
        (sht.Y_synthesis @ np.random.randn(L_max**2, B)).T,
        dtype=torch.float32)

    # ── SphericalCNN v1 ────────────────────────────────────────
    v1 = SphericalCNN(sht, channels=C, n_layers=2)
    y1 = v1(x)
    p1 = sum(p.numel() for p in v1.parameters())
    print(f'v1  output: {y1.shape}  params: {p1:,}')

    # ── SphericalCNNv2 ─────────────────────────────────────────
    print('v2 初期化中（exact Gaunt 計算中）...')
    v2 = SphericalCNNv2(sht, channels=C, n_layers=2, exact_gaunt=True)
    y2 = v2(x)
    p2 = sum(p.numel() for p in v2.parameters())
    print(f'v2  output: {y2.shape}  params: {p2:,}')

    # ── SO(3) 不変性チェック（z 軸回転） ──────────────────────
    N_PHI = n_phi
    k = N_PHI // 4  # 90°
    x_rot = torch.tensor(
        np.roll(x.numpy().reshape(B, n_theta, N_PHI), k, axis=2).reshape(B, -1),
        dtype=torch.float32)

    with torch.no_grad():
        err_v1 = (v1(x) - v1(x_rot)).abs().mean().item()
        err_v2 = (v2(x) - v2(x_rot)).abs().mean().item()

    print(f'\n90° 回転不変性エラー:')
    print(f'  v1 (NormNonlin): {err_v1:.2e}')
    print(f'  v2 (CGProduct) : {err_v2:.2e}')

    # ── パラメータ数の内訳 ─────────────────────────────────────
    print(f'\nパラメータ比較 (L_max={L_max}, channels={C}, n_layers=2):')
    print(f'  v1: {p1:,}  (S2Linear + NormNonlin)')
    print(f'  v2: {p2:,}  (S2Linear + CGProduct)')
