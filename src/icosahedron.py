"""
正二十面体メッシュ + 局所フレーム + 接続角（ゲージ場）

Cohen et al. (2019) のゲージ CNN の基盤となる幾何学的データを構築する。

────────────────────────────────────────────────────────────────
ゲージ理論の言語での設定
────────────────────────────────────────────────────────────────
・多様体 M = 正二十面体（12 頂点・20 面・30 辺）
・ゲージ群 G = U(1)
・各頂点 v に局所フレーム (e1_v, e2_v) を定める（ゲージを固定する）
・辺 (v→w) の接続角 α_{vw}: v の局所フレームで見た w の方位角
・ゲージ変換: フレームを φ_v だけ回転 → α_{vw} → α_{vw} - φ_v
"""

import numpy as np


# ============================================================
# 正二十面体の頂点・面・辺
# ============================================================

def build_icosahedron():
    """
    正二十面体の幾何データを構築する。

    黄金比 φ = (1+√5)/2 を使った標準構成:
      頂点 = (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) の 12 点
      （単位球面に正規化済み）

    Returns:
      vertices : (12, 3)  単位球面上の頂点座標
      faces    : (20, 3)  面ごとの頂点インデックス（外向き法線規則）
      edges    : (30, 2)  辺ごとの頂点インデックス対（無向辺）
    """
    phi = (1 + np.sqrt(5)) / 2

    # 12 頂点（正規化前）
    verts = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi, 0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [-phi, 0,  1], [ phi, 0, -1], [-phi, 0, -1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # 20 面（頂点 0 を中心とする 5 面 + 帯 + 頂点 3 を中心とする 5 面）
    # 頂点インデックスの対応:
    #  0=(0,1,φ), 1=(0,-1,φ), 2=(0,1,-φ), 3=(0,-1,-φ)
    #  4=(1,φ,0), 5=(-1,φ,0), 6=(1,-φ,0), 7=(-1,-φ,0)
    #  8=(φ,0,1), 9=(-φ,0,1), 10=(φ,0,-1), 11=(-φ,0,-1)
    faces = np.array([
        # 上側の 5 面（頂点 0 に接する）
        [ 0,  4,  5], [ 0,  5,  9], [ 0,  9,  1], [ 0,  1,  8], [ 0,  8,  4],
        # 中帯上側
        [ 4,  2,  5], [ 5,  2, 11], [ 5, 11,  9], [ 9, 11,  7], [ 9,  7,  1],
        # 中帯下側
        [ 1,  7,  6], [ 1,  6,  8], [ 8,  6, 10], [ 8, 10,  4], [ 4, 10,  2],
        # 下側の 5 面（頂点 3 に接する）
        [ 3,  6,  7], [ 3,  7, 11], [ 3, 11,  2], [ 3,  2, 10], [ 3, 10,  6],
    ], dtype=np.int64)

    # 辺: 各面の 3 辺を集合で重複除去
    edge_set = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([int(f[i]), int(f[(i + 1) % 3])]))
            edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int64)

    return verts, faces, edges


def validate_icosahedron(verts, faces, edges):
    """正二十面体の基本的な幾何的性質を検証する。"""
    V, F, E = len(verts), len(faces), len(edges)
    assert V == 12, f"頂点数 {V} ≠ 12"
    assert F == 20, f"面数 {F} ≠ 20"
    assert E == 30, f"辺数 {E} ≠ 30"

    # オイラー標数: V - E + F = 2 (球面位相)
    assert V - E + F == 2, f"オイラー標数 {V-E+F} ≠ 2"

    # 単位球面上に乗っているか
    norms = np.linalg.norm(verts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10), "頂点が単位球面上にない"

    # 全頂点の次数 = 5
    deg = np.zeros(V, dtype=int)
    for v, w in edges:
        deg[v] += 1
        deg[w] += 1
    assert np.all(deg == 5), f"次数が 5 でない頂点: {np.where(deg != 5)[0]}"

    # 辺長が一定（正二十面体の特徴）
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    assert np.allclose(lengths, lengths[0], atol=1e-10), "辺長が不均一"

    print(f"正二十面体の検証完了: V={V}, E={E}, F={F}, 辺長={lengths[0]:.6f}")
    return True


# ============================================================
# 局所フレームの構築（ゲージ選択）
# ============================================================

def compute_local_frames(vertices: np.ndarray) -> np.ndarray:
    """
    各頂点での接平面の正規直交フレーム (e1_v, e2_v) を計算する。

    ゲージ選択の規約:
      e1_v = n_v × ẑ を正規化（東向き接ベクトル）
      e2_v = n_v × e1_v（南北方向の接ベクトル）

      極点（n_v ∥ ẑ）では ẑ → x̂ にフォールバック。

    このフレーム選択が「ゲージを固定する」操作に相当する。
    別のフレームを選ぶ = ゲージ変換を施す。

    Returns: frames shape (V, 2, 3)
      frames[v, 0] = e1_v
      frames[v, 1] = e2_v
    """
    V = len(vertices)
    frames = np.zeros((V, 2, 3))
    z_hat = np.array([0.0, 0.0, 1.0])
    x_hat = np.array([1.0, 0.0, 0.0])

    for v, n in enumerate(vertices):
        e1 = np.cross(n, z_hat)
        if np.linalg.norm(e1) < 1e-8:
            e1 = np.cross(n, x_hat)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)
        frames[v, 0] = e1
        frames[v, 1] = e2

    return frames


# ============================================================
# 接続角の計算（ゲージ場）
# ============================================================

def compute_connection_angles(vertices: np.ndarray,
                               edges: np.ndarray,
                               frames: np.ndarray) -> np.ndarray:
    """
    各有向辺 (v→w) の接続角 α_{vw} を計算する。

    α_{vw} = 頂点 v の局所フレームにおける辺 v→w の方位角。

    計算手順:
      1. 辺方向: d = w - v
      2. v の接平面に射影: d_⊥ = d - (d·n_v) n_v
      3. 局所フレーム成分: x = d_⊥ · e1_v, y = d_⊥ · e2_v
      4. 接続角: α_{vw} = atan2(y, x)

    ゲージ変換の影響:
      フレームを φ_v 回転すると α_{vw} → α_{vw} - φ_v

    Returns: angles shape (E, 2)
      angles[e, 0] = α_{v→w}
      angles[e, 1] = α_{w→v}
    """
    E = len(edges)
    angles = np.zeros((E, 2))

    for e_idx, (v, w) in enumerate(edges):
        for direction, (src, dst) in enumerate([(v, w), (w, v)]):
            n  = vertices[src]
            e1 = frames[src, 0]
            e2 = frames[src, 1]
            d  = vertices[dst] - vertices[src]
            d_tan = d - np.dot(d, n) * n   # 接平面への射影
            x = np.dot(d_tan, e1)
            y = np.dot(d_tan, e2)
            angles[e_idx, direction] = np.arctan2(y, x)

    return angles


def compute_plaquette_holonomy(faces: np.ndarray,
                               edges: np.ndarray,
                               angles: np.ndarray) -> np.ndarray:
    """
    各三角面（プラケット）のホロノミー Θ_f を計算する。

    格子ゲージ理論（Harlow QFT3 §8.1, eq.8.35）の Wilson プラケットの U(1) 版。
    面を一周したときのフレームの累積回転 = ゲージ場の曲率（場の強さ F_μν の積分）。

    ────────────────────────────────────────────────────────────
    なぜ単純な接続角の和ではダメか
    ────────────────────────────────────────────────────────────
    α_{vw} は各頂点 v の「局所フレーム基準」で測った方位角。
    異なる頂点では基準フレームが違うので、α をそのまま足してもゲージ不変でない。
    辺 v→w を渡るときのフレームのねじれ（並行移動）は
        twist(v,w) = wrap(α_{wv} - α_{vw} + π)
    で与えられる（w から見た v 方向と v から見た w 方向は逆向き=+π、
    その差が局所フレーム間の相対回転）。これを一周足すと曲率になる。

    ゲージ不変性:
      フレームを φ_v 回転すると α_{vw} → α_{vw} - φ_v, α_{wv} → α_{wv} - φ_w。
      twist(v,w) = (α_{wv}-φ_w) - (α_{vw}-φ_v) + π。
      一周 v0→v1→v2→v0 で足すと φ の寄与が打ち消し合い、Θ_f はゲージ不変。

    Returns: holonomy shape (F,)  各面のホロノミー Θ_f ∈ (-π, π]
      正二十面体（細分なし）では全面 Θ_f = 36°（= 球面過剰角 = 4π/20）。
    """
    # 有向辺 → 接続角の辞書
    amap = {}
    for e_idx, (v, w) in enumerate(edges):
        amap[(int(v), int(w))] = float(angles[e_idx, 0])  # v→w
        amap[(int(w), int(v))] = float(angles[e_idx, 1])  # w→v

    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def twist(s, d):
        # 辺 s→d を渡るときの局所フレームの相対回転
        return wrap(amap[(d, s)] - amap[(s, d)] + np.pi)

    F = len(faces)
    holonomy = np.zeros(F, dtype=np.float64)
    for fi, (a, b, c) in enumerate(faces):
        a, b, c = int(a), int(b), int(c)
        holonomy[fi] = wrap(twist(a, b) + twist(b, c) + twist(c, a))

    return holonomy


def scatter_holonomy_to_vertices(faces: np.ndarray,
                                 holonomy: np.ndarray,
                                 num_vertices: int) -> np.ndarray:
    """
    面ごとのホロノミー Θ_f を、各頂点に集約する（頂点特徴量にするため）。

    頂点 v の曲率特徴 = v を含む面の Θ_f の平均（隣接プラケットの平均曲率）。
    GaugeCNN の入力は頂点ベースなので、面の量を頂点に落とす必要がある。

    Returns: vertex_curv shape (V, 2)  [cos Θ̄_v, sin Θ̄_v]
      ゲージ不変スカラーとして type-0 特徴量に concat できる。
    """
    V = num_vertices
    acc_cos = np.zeros(V)
    acc_sin = np.zeros(V)
    count   = np.zeros(V)
    for fi, (a, b, c) in enumerate(faces):
        for v in (int(a), int(b), int(c)):
            acc_cos[v] += np.cos(holonomy[fi])
            acc_sin[v] += np.sin(holonomy[fi])
            count[v]   += 1
    count = np.maximum(count, 1)
    return np.stack([acc_cos / count, acc_sin / count], axis=1).astype(np.float32)


def build_adjacency(vertices: np.ndarray,
                     edges: np.ndarray,
                     angles: np.ndarray) -> list:
    """
    各頂点の隣接リストを構築する。

    Returns: list of list
      adj[v] = [(w, alpha_vw), ...]  （w: 隣接頂点, alpha_vw: 接続角）
    """
    V = len(vertices)
    adj = [[] for _ in range(V)]
    for e_idx, (v, w) in enumerate(edges):
        adj[v].append((int(w), float(angles[e_idx, 0])))  # v→w
        adj[w].append((int(v), float(angles[e_idx, 1])))  # w→v
    return adj


# ============================================================
# 正二十面体の細分化（Subdivision）
# ============================================================

def _subdivide_once(verts: np.ndarray, faces: np.ndarray):
    """
    各三角形を4つの小三角形に分割する（1ステップ）。
    追加頂点は辺の中点を球面に投影。
    Level n: 10·4^n + 2 頂点
    """
    edge_map = {}
    new_verts = list(verts)
    new_faces = []

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_map:
            mid = (new_verts[i] + new_verts[j]) / 2
            mid = mid / np.linalg.norm(mid)   # 球面に投影
            edge_map[key] = len(new_verts)
            new_verts.append(mid)
        return edge_map[key]

    for f in faces:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces += [
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20],
        ]

    return np.array(new_verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


def subdivide(verts: np.ndarray, faces: np.ndarray, level: int = 1):
    """
    正二十面体を level 回細分化する。

    頂点数:
      level 0: 12
      level 1: 42
      level 2: 162
      level 3: 642
      level 4: 2562
      level 5: 10242  (解像度 ~2°)
    """
    for _ in range(level):
        verts, faces = _subdivide_once(verts, faces)
    return verts, faces


# ============================================================
# 汎用ゲージ構造データ（Subdivision 後のメッシュ対応）
# ============================================================

def build_gauge_data_general(verts: np.ndarray, faces: np.ndarray, device=None):
    """
    任意の三角メッシュに対してゲージ構造データを構築する。

    Subdivision 後のメッシュでは頂点次数が不均一になるため、
    max_deg でゼロパディングし nb_mask[v, k]=0 の位置を無効化する。

    Returns:
      nb_idx  : (V, max_deg)  隣接頂点インデックス（padding = 0）
      nb_ang  : (V, max_deg)  接続角（padding = 0）
      nb_mask : (V, max_deg)  有効フラグ（1.0=有効, 0.0=padding）
      verts   : (V, 3) ndarray
      faces   : (F, 3) ndarray
    """
    import torch

    frames = compute_local_frames(verts)

    # 辺を再構築
    edge_set = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([int(f[i]), int(f[(i + 1) % 3])]))
            edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int64)

    angles = compute_connection_angles(verts, edges, frames)
    adj    = build_adjacency(verts, edges, angles)

    V       = len(verts)
    max_deg = max(len(nbrs) for nbrs in adj)

    nb_idx_np  = np.zeros((V, max_deg), dtype=np.int64)
    nb_ang_np  = np.zeros((V, max_deg), dtype=np.float32)
    nb_mask_np = np.zeros((V, max_deg), dtype=np.float32)

    for v, nbrs in enumerate(adj):
        for k, (w, alpha) in enumerate(nbrs):
            nb_idx_np[v, k]  = w
            nb_ang_np[v, k]  = alpha
            nb_mask_np[v, k] = 1.0

    nb_idx  = torch.tensor(nb_idx_np,  dtype=torch.long)
    nb_ang  = torch.tensor(nb_ang_np,  dtype=torch.float32)
    nb_mask = torch.tensor(nb_mask_np, dtype=torch.float32)

    if device is not None:
        nb_idx  = nb_idx.to(device)
        nb_ang  = nb_ang.to(device)
        nb_mask = nb_mask.to(device)

    return nb_idx, nb_ang, nb_mask, verts, faces


# ============================================================
# 動作確認
# ============================================================

if __name__ == '__main__':
    verts, faces, edges = build_icosahedron()
    validate_icosahedron(verts, faces, edges)

    frames = compute_local_frames(verts)
    angles = compute_connection_angles(verts, edges, frames)
    adj    = build_adjacency(verts, edges, angles)

    print(f"\n局所フレーム (最初の 3 頂点):")
    for v in range(3):
        print(f"  v={v}: e1={frames[v,0].round(3)}, e2={frames[v,1].round(3)}")

    print(f"\n接続角 α_{{v→w}} (最初の 5 辺):")
    for e_idx in range(5):
        v, w = edges[e_idx]
        print(f"  ({v}→{w}): α={np.degrees(angles[e_idx,0]):.1f}°  "
              f"  ({w}→{v}): α={np.degrees(angles[e_idx,1]):.1f}°")

    print(f"\n隣接リスト（頂点 0 の 5 近傍）:")
    for w, alpha in adj[0]:
        print(f"  → v={w}: α={np.degrees(alpha):.1f}°")
