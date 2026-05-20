"""
ERA5 データを正二十面体メッシュ頂点へリサンプリングする。

緯度経度グリッド → 球面三角メッシュ頂点への補間は
scipy.interpolate.RegularGridInterpolator (linear) を使用。

データ取得方法:
  1. cdsapi（登録が必要: https://cds.climate.copernicus.eu/）
  2. ローカルの NetCDF ファイルを直接読み込む
"""

import numpy as np


def xyz_to_latlon(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    単位球面上の xyz 座標を緯度・経度（度）に変換する。

    Args:
        verts: (V, 3) 単位球面上の頂点座標

    Returns:
        lats: (V,)  緯度 [-90, 90]（度）
        lons: (V,)  経度 [0, 360)（度）
    """
    lats = np.degrees(np.arcsin(np.clip(verts[:, 2], -1.0, 1.0)))
    lons = np.degrees(np.arctan2(verts[:, 1], verts[:, 0])) % 360.0
    return lats, lons


def resample_to_mesh(field: np.ndarray,
                     src_lats: np.ndarray,
                     src_lons: np.ndarray,
                     tgt_verts: np.ndarray) -> np.ndarray:
    """
    緯度経度グリッドのスカラー場を三角メッシュ頂点に線形補間する。

    Args:
        field    : (nlat, nlon)  元の場（lat は降順でも昇順でも可）
        src_lats : (nlat,)  元のグリッドの緯度（昇順に並べ直す）
        src_lons : (nlon,)  元のグリッドの経度 [0, 360)
        tgt_verts: (V, 3)   補間先の頂点（単位球面上）

    Returns:
        values: (V,)  各頂点での補間値
    """
    from scipy.interpolate import RegularGridInterpolator

    # 緯度を昇順に揃える
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        field    = field[::-1, :]

    # 経度が [-180, 180] の場合は [0, 360) に変換
    if src_lons.min() < 0:
        src_lons = src_lons % 360.0
        order    = np.argsort(src_lons)
        src_lons = src_lons[order]
        field    = field[:, order]

    interp = RegularGridInterpolator(
        (src_lats, src_lons), field,
        method='linear', bounds_error=False, fill_value=None,
    )

    tgt_lats, tgt_lons = xyz_to_latlon(tgt_verts)
    query_pts = np.column_stack([tgt_lats, tgt_lons])
    return interp(query_pts)


def load_era5_nc(nc_path: str, variables: list[str],
                 verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ERA5 の NetCDF ファイルを読み込み、メッシュ頂点にリサンプリングする。

    Args:
        nc_path  : .nc ファイルのパス
        variables: 使用する変数名リスト（例: ['t2m', 'z']）
        verts    : (V, 3) メッシュ頂点

    Returns:
        features : (T, V, len(variables))  時系列特徴量行列
        times    : (T,)  時刻配列（numpy datetime64）
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    T = len(ds['time'])
    V = len(verts)
    C = len(variables)
    features = np.zeros((T, V, C), dtype=np.float32)

    for t in range(T):
        for c, var in enumerate(variables):
            field = ds[var].isel(time=t).values  # (nlat, nlon)
            if field.ndim == 3:   # pressure level あり
                field = field[0]  # 最初のレベルを使用
            features[t, :, c] = resample_to_mesh(field, lats, lons, verts)

    times = ds['time'].values
    ds.close()
    return features, times


def load_weatherbench2(verts: np.ndarray,
                       variables: list[str] = ['2m_temperature'],
                       start: str = '2020-01-01',
                       end: str   = '2020-12-31',
                       pressure_level: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    WeatherBench2 (Google Cloud Storage) から ERA5 データを登録なしで取得し、
    メッシュ頂点にリサンプリングする。

    必要パッケージ: gcsfs, zarr
      !pip install gcsfs zarr -q

    利用可能な単一レベル変数 (pressure_level=None):
      '2m_temperature', 'mean_sea_level_pressure',
      '10m_u_component_of_wind', '10m_v_component_of_wind',
      'total_precipitation_6hr'

    利用可能な気圧面変数 (pressure_level=500 など):
      'geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind',
      'specific_humidity', 'vertical_velocity'

    Args:
        verts          : (V, 3) メッシュ頂点（単位球面）
        variables      : 変数名リスト
        start / end    : 取得期間 'YYYY-MM-DD'
        pressure_level : 気圧面 (hPa)。None の場合は単一レベル変数。

    Returns:
        features : (T, V, C)  float32
        times    : (T,)  numpy datetime64
    """
    import xarray as xr

    try:
        import gcsfs
    except ImportError:
        raise ImportError("gcsfs が必要です: pip install gcsfs zarr")

    gcs = gcsfs.GCSFileSystem(token='anon')

    if pressure_level is not None:
        zarr_path = (
            'gs://weatherbench2/datasets/era5/'
            '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
        )
    else:
        zarr_path = (
            'gs://weatherbench2/datasets/era5/'
            '1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
        )

    print(f"WeatherBench2 に接続中...")
    store = gcs.get_mapper(zarr_path)
    try:
        ds = xr.open_zarr(store, consolidated=True)
    except Exception:
        ds = xr.open_zarr(store, consolidated=False)

    # 時間・気圧面でスライス
    ds = ds.sel(time=slice(start, end))
    if pressure_level is not None and 'level' in ds.dims:
        ds = ds.sel(level=pressure_level, method='nearest')

    lats = ds['latitude'].values   # (121,)  -90 → 90
    lons = ds['longitude'].values  # (240,)   0 → 359.5

    T = len(ds['time'])
    V = len(verts)
    C = len(variables)
    features = np.zeros((T, V, C), dtype=np.float32)

    for c, var in enumerate(variables):
        print(f"  変数 '{var}' をリサンプリング中 (T={T})...")
        da = ds[var]
        if da.dims[1] == 'longitude':  # (T, lon, lat) → (T, lat, lon)
            da = da.transpose('time', 'latitude', 'longitude')
        data = da.values  # (T, nlat, nlon)
        for t in range(T):
            features[t, :, c] = resample_to_mesh(data[t], lats, lons, verts)

    times = ds['time'].values
    ds.close()
    print(f"完了: shape={features.shape}")
    return features, times


def download_era5(start_date: str, end_date: str,
                  variables: list[str], save_path: str,
                  pressure_level: int | None = None):
    """
    cdsapi を使って ERA5 データをダウンロードする。

    事前に ~/.cdsapirc を設定する必要がある:
      url: https://cds.climate.copernicus.eu/api/v2
      key: <your-uid>:<your-api-key>

    Args:
        start_date    : 'YYYY-MM-DD'
        end_date      : 'YYYY-MM-DD'
        variables     : CDS 変数名リスト（例: ['2m_temperature', 'geopotential']）
        save_path     : 保存先 .nc ファイルパス
        pressure_level: 気圧面 (hPa)。None の場合は単一レベル変数として取得。
    """
    import cdsapi
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end   = datetime.strptime(end_date,   '%Y-%m-%d')
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)

    c = cdsapi.Client()
    request = {
        'product_type': 'reanalysis',
        'variable': variables,
        'date': dates,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'format': 'netcdf',
    }
    if pressure_level is not None:
        dataset = 'reanalysis-era5-pressure-levels'
        request['pressure_level'] = [str(pressure_level)]
    else:
        dataset = 'reanalysis-era5-single-levels'

    c.retrieve(dataset, request, save_path)
    print(f"ERA5 ダウンロード完了: {save_path}")


# ============================================================
# 合成データ生成（ERA5 なしで動作確認用）
# ============================================================

def make_synthetic_era5(verts: np.ndarray, T: int = 100,
                        n_vars: int = 3, seed: int = 42,
                        nb_idx: np.ndarray | None = None,
                        alpha: float = 0.4) -> np.ndarray:
    """
    ERA5 を模した合成球面場を生成する（空間相関付き）。

    nb_idx が与えられると、各ステップで隣接頂点の平均を拡散させる（DAR(1) モデル）。
    これにより隣接頂点の情報が予測に有用になり、GNN のメリットが現れる。

    Args:
        nb_idx : (V, max_deg) 隣接インデックス（numpy）。None の場合は空間独立 AR(1)。
        alpha  : 空間拡散の強さ（0=独立, 1=完全拡散）

    Returns:
        features: (T, V, n_vars)
    """
    rng  = np.random.default_rng(seed)
    V    = len(verts)
    lats, lons = xyz_to_latlon(verts)
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)

    patterns = np.column_stack([
        np.cos(lats_r),
        np.cos(2 * lats_r),
        np.cos(lats_r) * np.cos(lons_r),
        np.cos(2 * lats_r) * np.cos(2 * lons_r),
        np.sin(lats_r),
    ])  # (V, 5)

    features = np.zeros((T, V, n_vars), dtype=np.float32)
    phi = 0.9

    for c in range(n_vars):
        coeffs = rng.standard_normal(5)
        base   = patterns @ coeffs  # (V,)
        state  = base.copy()
        for t in range(T):
            noise = rng.standard_normal(V) * 0.2
            if nb_idx is not None:
                neighbor_vals = state[nb_idx]           # (V, max_deg)
                spatial_mean  = neighbor_vals.mean(axis=1)  # (V,)
                diffused = alpha * spatial_mean + (1 - alpha) * state
            else:
                diffused = state
            state = phi * diffused + (1 - phi) * base + noise
            features[t, :, c] = state.astype(np.float32)

    return features
