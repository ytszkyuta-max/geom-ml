# geom-ml — 幾何学的深層学習の探索

群論・微分幾何・ゲージ理論を機械学習に接続し、実世界（特に気候・地球物理学）への応用を探る研究プロジェクト。

## 数学的背景

- 群の表現論（吉川圭二「群と表現」）
- 曲線と曲面の微分幾何
- 接続の微分幾何・ゲージ理論

## ロードマップ

### Step 1 — 統一的枠組みの理解
- [ ] Bronstein et al. (2021) "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
  - https://geometricdeeplearning.com
  - 深層学習を群の対称性・同変写像として統一記述

### Step 2 — 球面CNN の数学と実装
- [ ] Cohen et al. (2018) "Spherical CNNs" (NeurIPS 2018)
  - S² 上の畳み込み = SO(3) のフーリエ変換
  - Wigner D行列・球面調和関数を使った実装
- [ ] 球面調和変換を自前実装して理解を確かめる

### Step 3 — ゲージ同変 CNN
- [ ] Cohen et al. (2019) "Gauge Equivariant Convolutional Networks and the Icosahedral CNN" (ICML 2019)
  - 主束・接続・ゲージ変換が直接登場する論文
  - 読了済みのゲージ理論の言語でほぼ書かれている

### Step 4 — ERA5 データへの適用
- [ ] ERA5（ECMWF）から気温・気圧データを取得
- [ ] Step 2〜3 のモデルで球面上の予測タスクを解く

## 使用ライブラリ

```
healpy   # HEALPix 球面グリッド
e3nn     # SO(3) 同変NN（PyTorch）
s2fft    # 球面フーリエ変換（JAX）
cdsapi   # ERA5 データ取得
```

## ディレクトリ構成

```
geom-ml/
├── README.md
├── notes/      # 論文読書メモ・数学ノート
├── papers/     # 論文リファレンス
└── src/        # 実装
```
