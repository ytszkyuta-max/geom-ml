"""
ゲージ同変 CNN on 正二十面体 (Cohen et al. 2019)

U(1) ゲージ群の主束 P → 正二十面体 M を基盤に、
ゲージ同変メッセージパッシングを実装する。

────────────────────────────────────────────────────────────────
理論的背景
────────────────────────────────────────────────────────────────
・型 n の特徴量 f_v[n]: ゲージ変換 φ_v → e^{-inφ_v} · f_v[n]
・GaugeConv (type-0 → type-n):
    f'_v[n, c_out] = Σ_{c_in} W_n[c_out,c_in] · Σ_{w∈N(v)} e^{inα_{vw}} · f_w[c_in]
  ゲージ同変性: α_{vw} → α_{vw} - φ_v ⇒ f'_v[n] → e^{-inφ_v} · f'_v[n]
・GaugeConvTypeN (type-n → type-n):
    f'_v[n, c_out] = Σ_{c_in} W_n[c_out,c_in] · Σ_{w} e^{inα_{vw}} · f_w[n, c_in]
  （チャネルを混ぜるが型 n を保つ）
・GaugeNorm: ‖f_v[n]‖ のみ変換（ゲージ同変非線形性）
・InvariantPool: ‖f_v[n]‖² → type-0 スカラー（ゲージ不変）
"""

import numpy as np
import torch
import torch.nn as nn
from icosahedron import (build_icosahedron, compute_local_frames,
                         compute_connection_angles, build_adjacency)


# ============================================================
# ゲージ構造データの構築
# ============================================================

def build_gauge_data(device=None):
    """
    正二十面体のゲージ構造データをTorchテンソルとして構築する。

    Returns:
      neighbor_idx    : (12, 5) 長 Tensor  各頂点の隣接頂点インデックス
      neighbor_angles : (12, 5) float Tensor  接続角 α_{v→w}
      verts           : (12, 3) ndarray  頂点座標（単位球面）
      faces           : (20, 3) ndarray  面インデックス
    """
    verts, faces, edges = build_icosahedron()
    frames = compute_local_frames(verts)
    angles = compute_connection_angles(verts, edges, frames)
    adj    = build_adjacency(verts, edges, angles)

    V = len(verts)
    nb_idx_np  = np.zeros((V, 5), dtype=np.int64)
    nb_ang_np  = np.zeros((V, 5), dtype=np.float32)

    for v in range(V):
        nbrs = adj[v]
        assert len(nbrs) == 5, f"頂点 {v} の次数が 5 でない: {len(nbrs)}"
        for k, (w, alpha) in enumerate(nbrs):
            nb_idx_np[v, k]  = w
            nb_ang_np[v, k]  = alpha

    nb_idx = torch.tensor(nb_idx_np, dtype=torch.long)
    nb_ang = torch.tensor(nb_ang_np, dtype=torch.float32)
    if device is not None:
        nb_idx = nb_idx.to(device)
        nb_ang = nb_ang.to(device)

    return nb_idx, nb_ang, verts, faces


# ============================================================
# GaugeConv: type-0 入力 → type-n 複素出力
# ============================================================

class GaugeConv(nn.Module):
    """
    U(1) ゲージ同変畳み込み層（type-0 スカラー入力 → type-n 複素出力）

    f'_v[n, c_out] = Σ_{c_in} W_n[c_out,c_in] · Σ_{w∈N(v)} e^{inα_{vw}} · f_w[c_in]

    ゲージ同変性の証明:
      α_{vw} → α_{vw} - φ_v のとき
      Σ_w e^{in(α_{vw}-φ_v)} f_w = e^{-inφ_v} · Σ_w e^{inα_{vw}} f_w
      ∴ f'_v[n] → e^{-inφ_v} · f'_v[n]  □

    Args:
        c_in     : 入力チャネル数（type-0 実数特徴量）
        c_out    : 各 type の出力チャネル数
        n_types  : 使用する Fourier 型 n ∈ {0, 1, ..., n_types-1}
        neighbor_idx    : (V, 5) 隣接頂点インデックス
        neighbor_angles : (V, 5) 接続角 α_{vw}

    入力:  (batch, V, c_in)           type-0 実数特徴量
    出力:  (batch, V, n_types, c_out, 2)  複素特徴量 [...,0]=実部 [...,1]=虚部
    """

    def __init__(self, c_in: int, c_out: int, n_types: int,
                 neighbor_idx: torch.Tensor, neighbor_angles: torch.Tensor):
        super().__init__()
        self.c_in    = c_in
        self.c_out   = c_out
        self.n_types = n_types

        self.register_buffer('neighbor_idx',    neighbor_idx)    # (V, 5)
        self.register_buffer('neighbor_angles', neighbor_angles) # (V, 5)

        # 複素重み W_n ∈ C^{c_out × c_in} を実部・虚部で保持
        scale = 1.0 / np.sqrt(c_in * 5)
        self.weight_real = nn.Parameter(torch.randn(n_types, c_out, c_in) * scale)
        self.weight_imag = nn.Parameter(torch.randn(n_types, c_out, c_in) * scale)

    def forward(self, x: torch.Tensor,
                nb_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, V, _ = x.shape
        K = self.neighbor_idx.shape[1]  # max_deg（通常5、subdivision後は6）

        # 隣接頂点の特徴量を収集: (B, V, K, c_in)
        idx_flat = self.neighbor_idx.reshape(-1)               # (V*K,)
        x_nb = x[:, idx_flat, :].reshape(B, V, K, self.c_in)  # (B,V,K,c_in)

        # padding 位置をゼロマスク: nb_mask (V, K) → (1,V,K,1)
        if nb_mask is not None:
            x_nb = x_nb * nb_mask[None, :, :, None]

        # Fourier 位相 e^{inα_{vw}}: ns (n_types,), angles (V,K) → (V,K,n_types)
        ns     = torch.arange(self.n_types, device=x.device, dtype=torch.float32)
        phases = ns[None, None, :] * self.neighbor_angles[:, :, None]  # (V,K,N)
        cos_p  = torch.cos(phases)  # (V,K,N)
        sin_p  = torch.sin(phases)

        # 隣接和: agg[v,n,c] = Σ_w e^{inα_{vw}} · f_w[c]
        agg_r = torch.einsum('bvkc,vkn->bvnc', x_nb, cos_p)   # (B,V,N,c_in)
        agg_i = torch.einsum('bvkc,vkn->bvnc', x_nb, sin_p)

        # 複素行列積: (W_r + iW_i)(A_r + iA_i)
        # = (W_r A_r - W_i A_i) + i(W_r A_i + W_i A_r)
        out_r = (torch.einsum('noc,bvnc->bvno', self.weight_real, agg_r)
               - torch.einsum('noc,bvnc->bvno', self.weight_imag, agg_i))
        out_i = (torch.einsum('noc,bvnc->bvno', self.weight_real, agg_i)
               + torch.einsum('noc,bvnc->bvno', self.weight_imag, agg_r))

        return torch.stack([out_r, out_i], dim=-1)  # (B,V,N,c_out,2)


# ============================================================
# GaugeConvTypeN: type-n → type-n（チャネル混合、型は保持）
# ============================================================

class GaugeConvTypeN(nn.Module):
    """
    型 n を保つゲージ同変畳み込み（type-n → type-n）

    各 type n について独立にメッセージパッシング:
    f'_v[n, c_out] = Σ_{c_in} W_n[c_out,c_in] · Σ_{w∈N(v)} e^{inα_{vw}} · f_w[n, c_in]

    複素入力 f_w[n] = (f_r + if_i) に位相 e^{inα_{vw}} = (cosθ + i sinθ) を掛ける:
    e^{inα} · f = (cosθ·f_r - sinθ·f_i) + i(sinθ·f_r + cosθ·f_i)

    入力:  (batch, V, n_types, c_in, 2)
    出力:  (batch, V, n_types, c_out, 2)
    """

    def __init__(self, c_in: int, c_out: int, n_types: int,
                 neighbor_idx: torch.Tensor, neighbor_angles: torch.Tensor):
        super().__init__()
        self.n_types = n_types

        self.register_buffer('neighbor_idx',    neighbor_idx)
        self.register_buffer('neighbor_angles', neighbor_angles)

        scale = 1.0 / np.sqrt(c_in * 5)
        self.weight_real = nn.Parameter(torch.randn(n_types, c_out, c_in) * scale)
        self.weight_imag = nn.Parameter(torch.randn(n_types, c_out, c_in) * scale)

    def forward(self, x: torch.Tensor,
                nb_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, V, N, C, _ = x.shape
        K = self.neighbor_idx.shape[1]  # max_deg

        # 隣接特徴量を収集
        idx_flat = self.neighbor_idx.reshape(-1)   # (V*K,)
        x_r = x[..., 0]  # (B,V,N,C)
        x_i = x[..., 1]
        nb_r = x_r[:, idx_flat, :, :].reshape(B, V, K, N, C)  # (B,V,K,N,C)
        nb_i = x_i[:, idx_flat, :, :].reshape(B, V, K, N, C)

        if nb_mask is not None:
            m = nb_mask[None, :, :, None, None]   # (1,V,K,1,1)
            nb_r = nb_r * m
            nb_i = nb_i * m

        # Fourier 位相: ns (N,), angles (V,K) → (V,K,N)
        ns     = torch.arange(N, device=x.device, dtype=torch.float32)
        phases = ns[None, None, :] * self.neighbor_angles[:, :, None]  # (V,K,N)
        cos_p  = torch.cos(phases)  # (V,K,N)
        sin_p  = torch.sin(phases)

        # e^{inα} · f_w[n]: broadcast (B,V,K,N,C) × (1,V,K,N,1)
        cos_b = cos_p[None, :, :, :, None]   # (1,V,K,N,1)
        sin_b = sin_p[None, :, :, :, None]
        trans_r = nb_r * cos_b - nb_i * sin_b   # (B,V,5,N,C)
        trans_i = nb_r * sin_b + nb_i * cos_b

        # 近傍の和: (B,V,N,C)
        agg_r = trans_r.sum(dim=2)
        agg_i = trans_i.sum(dim=2)

        # 複素行列積
        out_r = (torch.einsum('noc,bvnc->bvno', self.weight_real, agg_r)
               - torch.einsum('noc,bvnc->bvno', self.weight_imag, agg_i))
        out_i = (torch.einsum('noc,bvnc->bvno', self.weight_real, agg_i)
               + torch.einsum('noc,bvnc->bvno', self.weight_imag, agg_r))

        return torch.stack([out_r, out_i], dim=-1)  # (B,V,N,c_out,2)


# ============================================================
# GaugeNorm: ゲージ同変非線形性
# ============================================================

class GaugeNorm(nn.Module):
    """
    型 n ごとのノルムベース非線形性（ゲージ同変）。

    各 type n に対して: f_v[n] → f_v[n] · σ(‖f_v[n]‖ + b_n) / ‖f_v[n]‖
    ここで σ = ReLU。ノルムのみ変換し位相（方向）は保持する。

    ゲージ同変性: ‖e^{-inφ}·f‖ = ‖f‖ なのでスケール係数が不変 → 変換が保たれる。

    入力・出力: (batch, V, n_types, c_out, 2)
    """

    def __init__(self, n_types: int, c_out: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_types, c_out))  # ノルムのシフト

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # norm: (B,V,N,c_out)  各複素特徴量のノルム
        norm = x.norm(dim=-1)                                   # (B,V,N,c_out)
        bias = self.bias[None, None, :, :]                      # (1,1,N,c_out)
        new_norm = torch.relu(norm + bias)                      # (B,V,N,c_out)
        # ゼロ除算を防ぐ
        scale = new_norm / norm.clamp(min=1e-8)                 # (B,V,N,c_out)
        return x * scale.unsqueeze(-1)                          # (B,V,N,c_out,2)


# ============================================================
# InvariantPool: ゲージ不変スカラーへの変換
# ============================================================

class InvariantPool(nn.Module):
    """
    各型 n のノルム二乗を取ってゲージ不変スカラー特徴量を得る。

    ‖f_v[n]‖² = f_r² + f_i² はゲージ変換 e^{-inφ} に対して不変（|e^{iθ}|=1）。

    入力:  (batch, V, n_types, c_out, 2)
    出力:  (batch, V, n_types * c_out)  type-0 ゲージ不変スカラー
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, N, C, _ = x.shape
        norm_sq = (x ** 2).sum(dim=-1)          # (B,V,N,C)
        return norm_sq.reshape(B, V, N * C)     # (B,V,N*C)


# ============================================================
# IcosGaugeCNN: 正二十面体ゲージ CNN
# ============================================================

class IcosGaugeCNN(nn.Module):
    """
    正二十面体ゲージ CNN の 2 層実装。

    アーキテクチャ:
      入力 (type-0) →
      GaugeConv → GaugeNorm → InvariantPool (type-0) →
      GaugeConv → GaugeNorm → InvariantPool (type-0) →
      Linear → 出力

    InvariantPool のおかげで出力はゲージ不変（任意の局所フレーム回転に対して不変）。

    Args:
        c_in      : 入力チャネル数（各頂点のスカラー特徴量次元）
        c_hidden  : 中間チャネル数
        n_types   : Fourier 型数 {0,...,n_types-1}
        n_out     : 出力次元（各頂点）
        neighbor_idx    : (12, 5) バッファ
        neighbor_angles : (12, 5) バッファ
    """

    def __init__(self, c_in: int, c_hidden: int, n_types: int, n_out: int,
                 neighbor_idx: torch.Tensor, neighbor_angles: torch.Tensor):
        super().__init__()
        self.conv1  = GaugeConv(c_in,              c_hidden, n_types, neighbor_idx, neighbor_angles)
        self.norm1  = GaugeNorm(n_types, c_hidden)
        self.pool1  = InvariantPool()

        inv_dim = n_types * c_hidden
        self.conv2  = GaugeConv(inv_dim,            c_hidden, n_types, neighbor_idx, neighbor_angles)
        self.norm2  = GaugeNorm(n_types, c_hidden)
        self.pool2  = InvariantPool()

        self.head   = nn.Linear(n_types * c_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, V=12, c_in)  type-0 入力特徴量
        Returns:
            out: (batch, V=12, n_out)  ゲージ不変出力
        """
        x = self.pool1(self.norm1(self.conv1(x)))   # (B,V, n_types*c_hidden) — gauge-invariant
        x = self.pool2(self.norm2(self.conv2(x)))   # (B,V, n_types*c_hidden)
        return self.head(x)                          # (B,V, n_out)


# ============================================================
# IcosGaugeCNNGeneral: Subdivision 後のメッシュに対応した汎用版
# ============================================================

class IcosGaugeCNNGeneral(nn.Module):
    """
    任意の細分化レベルの正二十面体メッシュに対応するゲージ CNN。

    build_gauge_data_general() の出力を受け取り、
    nb_mask で padding 位置を無効化する。

    Args:
        c_in      : 入力チャネル数（各頂点のスカラー特徴量数）
        c_hidden  : 中間チャネル数
        n_types   : Fourier 型数
        n_out     : 出力次元
        nb_idx    : (V, max_deg)
        nb_ang    : (V, max_deg)
        nb_mask   : (V, max_deg)  padding マスク
    """

    def __init__(self, c_in: int, c_hidden: int, n_types: int, n_out: int,
                 nb_idx: torch.Tensor, nb_ang: torch.Tensor, nb_mask: torch.Tensor):
        super().__init__()
        self.register_buffer('nb_mask', nb_mask)

        inv_dim = n_types * c_hidden

        self.conv1 = GaugeConv(c_in,    c_hidden, n_types, nb_idx, nb_ang)
        self.norm1 = GaugeNorm(n_types, c_hidden)
        self.pool1 = InvariantPool()
        self.ln1   = nn.LayerNorm(inv_dim)  # InvariantPool の二乗でスケールが爆発しないよう正規化

        self.conv2 = GaugeConv(inv_dim, c_hidden, n_types, nb_idx, nb_ang)
        self.norm2 = GaugeNorm(n_types, c_hidden)
        self.pool2 = InvariantPool()
        self.ln2   = nn.LayerNorm(inv_dim)

        self.head  = nn.Linear(inv_dim, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, V, c_in)
        Returns:
            (batch, V, n_out)
        """
        x = self.ln1(self.pool1(self.norm1(self.conv1(x, self.nb_mask))))
        x = self.ln2(self.pool2(self.norm2(self.conv2(x, self.nb_mask))))
        return self.head(x)


# ============================================================
# 動作確認
# ============================================================

if __name__ == '__main__':
    device = torch.device('cpu')
    nb_idx, nb_ang, verts, faces = build_gauge_data(device)

    B, V, C_IN, C_HID, N_TYPES = 2, 12, 4, 8, 4

    print("=== GaugeConv ===")
    conv = GaugeConv(C_IN, C_HID, N_TYPES, nb_idx, nb_ang)
    x    = torch.randn(B, V, C_IN)
    out  = conv(x)
    print(f"  入力: {tuple(x.shape)} → 出力: {tuple(out.shape)}")
    print(f"  期待: (B={B}, V={V}, n_types={N_TYPES}, c_out={C_HID}, 2)")

    print("\n=== GaugeConvTypeN ===")
    conv_n = GaugeConvTypeN(C_HID, C_HID, N_TYPES, nb_idx, nb_ang)
    out_n  = conv_n(out)
    print(f"  入力: {tuple(out.shape)} → 出力: {tuple(out_n.shape)}")

    print("\n=== IcosGaugeCNN ===")
    model = IcosGaugeCNN(C_IN, C_HID, N_TYPES, n_out=1, neighbor_idx=nb_idx, neighbor_angles=nb_ang)
    y     = model(x)
    print(f"  入力: {tuple(x.shape)} → 出力: {tuple(y.shape)}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {params}")
