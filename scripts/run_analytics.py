"""
GPU-Accelerated Spatial Analytics Pipeline v2
Implements ArcGIS-equivalent spatial analysis with NVIDIA GPU acceleration.

Analytics implemented:
  - Zone-level aggregation (pickup/dropoff/revenue/tips)
  - GPU-accelerated Kernel Density Estimation (ArcGIS Kernel Density equivalent)
  - GPU-accelerated Getis-Ord Gi* Hot Spot Analysis (ArcGIS Hot Spot Analysis equivalent)
  - GPU-accelerated Moran's I Spatial Autocorrelation (ArcGIS Spatial Autocorrelation)
  - GPU-accelerated spatial weights matrix construction
  - H3 hexagonal spatial binning (Uber H3)
  - DBSCAN spatial clustering
  - OD flow matrix with GPU pairwise distances
  - Temporal pattern analysis
  - GPU-accelerated inverse distance weighting interpolation
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from scipy import stats as scipy_stats
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import cupy as cp
    GPU = True
    print(f"GPU acceleration enabled — CuPy {cp.__version__}")
    print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
except ImportError:
    GPU = False
    print("CuPy not available — falling back to CPU (NumPy)")

try:
    from libpysal.weights import Queen, KNN
    from esda.getisord import G_Local
    from esda.moran import Moran
    PYSAL = True
    print("PySAL spatial statistics enabled (Getis-Ord Gi*, Moran's I)")
except ImportError:
    PYSAL = False
    print("PySAL not available — hot spot analysis will use GPU fallback")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
WEBAPP_DATA = os.path.join(BASE, "webapp", "data")
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(WEBAPP_DATA, exist_ok=True)

def save_geojson(gdf_or_dict, name):
    """Save to both output/ and webapp/data/."""
    for dest in [OUTPUT, WEBAPP_DATA]:
        path = os.path.join(dest, name)
        if isinstance(gdf_or_dict, gpd.GeoDataFrame):
            gdf_or_dict.to_file(path, driver="GeoJSON")
        else:
            with open(path, "w") as f:
                json.dump(gdf_or_dict, f)

def save_json(data, name):
    for dest in [OUTPUT, WEBAPP_DATA]:
        with open(os.path.join(dest, name), "w") as f:
            json.dump(data, f, indent=2 if len(json.dumps(data)) < 10000 else None)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n{'='*70}")
print("  LOADING DATA")
print("{'='*70}")
t0 = time.time()

zones = gpd.read_file(os.path.join(DATA, "taxi_zones", "taxi_zones", "taxi_zones.shp"))
zones = zones.to_crs(epsg=4326)
zones["centroid_x"] = zones.geometry.centroid.x
zones["centroid_y"] = zones.geometry.centroid.y
print(f"  Loaded {len(zones)} taxi zones")

trips = pd.read_parquet(os.path.join(DATA, "yellow_tripdata_2024-01.parquet"))
print(f"  Loaded {len(trips):,} taxi trips ({time.time()-t0:.1f}s)")

valid_ids = set(zones["LocationID"].values)
trips = trips[
    trips["PULocationID"].isin(valid_ids) &
    trips["DOLocationID"].isin(valid_ids) &
    (trips["total_amount"] > 0) &
    (trips["trip_distance"] > 0) &
    (trips["trip_distance"] < 100)
].copy()
trips["duration_min"] = (
    (trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"])
    .dt.total_seconds() / 60
)
trips["pickup_hour"] = trips["tpep_pickup_datetime"].dt.hour
trips["pickup_dow"] = trips["tpep_pickup_datetime"].dt.dayofweek
print(f"  After cleaning: {len(trips):,} valid trips")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ZONE-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  ZONE-LEVEL AGGREGATION")
print(f"{'='*70}")
t0 = time.time()

pu_stats = trips.groupby("PULocationID").agg(
    trip_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_tip=("tip_amount", "mean"),
    avg_distance=("trip_distance", "mean"),
    avg_total=("total_amount", "mean"),
    total_revenue=("total_amount", "sum"),
    avg_passengers=("passenger_count", "mean"),
    avg_duration=("duration_min", "mean"),
    med_fare=("fare_amount", "median"),
).reset_index().rename(columns={"PULocationID": "LocationID"})

do_stats = trips.groupby("DOLocationID").agg(
    dropoff_count=("VendorID", "count"),
    avg_fare_do=("fare_amount", "mean"),
).reset_index().rename(columns={"DOLocationID": "LocationID"})

for col in pu_stats.select_dtypes(include="float64").columns:
    pu_stats[col] = pu_stats[col].round(2)
for col in do_stats.select_dtypes(include="float64").columns:
    do_stats[col] = do_stats[col].round(2)

zones_pu = zones.merge(pu_stats, on="LocationID", how="left").fillna(0)
zones_do = zones.merge(do_stats, on="LocationID", how="left").fillna(0)

# Compute pickup/dropoff ratio (demand imbalance)
combined = pu_stats[["LocationID", "trip_count"]].merge(
    do_stats[["LocationID", "dropoff_count"]], on="LocationID", how="outer"
).fillna(0)
combined["pu_do_ratio"] = (combined["trip_count"] / combined["dropoff_count"].replace(0, 1)).round(2)
zones_pu = zones_pu.merge(combined[["LocationID", "pu_do_ratio"]], on="LocationID", how="left").fillna(0)

zones_pu["geometry"] = zones_pu["geometry"].simplify(0.0005, preserve_topology=True)
zones_do["geometry"] = zones_do["geometry"].simplify(0.0005, preserve_topology=True)

save_geojson(zones_pu, "zone_pickups.geojson")
save_geojson(zones_do, "zone_dropoffs.geojson")
print(f"  Zone aggregation complete ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. GPU-ACCELERATED SPATIAL WEIGHTS MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GPU-ACCELERATED SPATIAL WEIGHTS & DISTANCE MATRIX")
print(f"{'='*70}")
t0 = time.time()

cx = zones["centroid_x"].values.astype(np.float64)
cy = zones["centroid_y"].values.astype(np.float64)
n = len(zones)

if GPU:
    cx_gpu = cp.asarray(cx)
    cy_gpu = cp.asarray(cy)

    # Haversine-approximated distance in km (GPU accelerated)
    dx_km = (cx_gpu[:, None] - cx_gpu[None, :]) * 85.39  # cos(40.7°) * 111.32
    dy_km = (cy_gpu[:, None] - cy_gpu[None, :]) * 111.32
    dist_matrix_gpu = cp.sqrt(dx_km**2 + dy_km**2)
    dist_matrix = cp.asnumpy(dist_matrix_gpu)

    # GPU-accelerated inverse distance weights (for IDW interpolation & Gi*)
    # W_ij = 1/d_ij for d < threshold, 0 otherwise
    threshold_km = 5.0
    inv_dist_gpu = cp.where(
        (dist_matrix_gpu > 0) & (dist_matrix_gpu < threshold_km),
        1.0 / dist_matrix_gpu,
        0.0
    )
    # Row-standardize
    row_sums = inv_dist_gpu.sum(axis=1, keepdims=True)
    row_sums = cp.where(row_sums == 0, 1.0, row_sums)
    w_matrix_gpu = inv_dist_gpu / row_sums
    w_matrix = cp.asnumpy(w_matrix_gpu)
    method = "GPU (CuPy)"
else:
    dx_km = (cx[:, None] - cx[None, :]) * 85.39
    dy_km = (cy[:, None] - cy[None, :]) * 111.32
    dist_matrix = np.sqrt(dx_km**2 + dy_km**2)
    threshold_km = 5.0
    inv_dist = np.where((dist_matrix > 0) & (dist_matrix < threshold_km), 1.0 / dist_matrix, 0.0)
    row_sums = inv_dist.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    w_matrix = inv_dist / row_sums
    method = "CPU (NumPy)"

print(f"  {n}x{n} distance matrix via {method}")
print(f"  Mean distance: {dist_matrix.mean():.2f} km | Max: {dist_matrix.max():.2f} km")
print(f"  Spatial weights: {threshold_km} km threshold, inverse-distance ({time.time()-t0:.2f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GPU-ACCELERATED GETIS-ORD Gi* HOT SPOT ANALYSIS
#    (Equivalent to ArcGIS Pro "Hot Spot Analysis" / "Optimized Hot Spot Analysis")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GPU-ACCELERATED GETIS-ORD Gi* HOT SPOT ANALYSIS")
print(f"{'='*70}")
t0 = time.time()

trip_counts = zones_pu.set_index("LocationID")["trip_count"].reindex(zones["LocationID"]).fillna(0).values

if GPU:
    x = cp.asarray(trip_counts, dtype=cp.float64)
    W = cp.asarray(w_matrix, dtype=cp.float64)

    x_mean = cp.mean(x)
    n_f = cp.float64(n)
    S = cp.sqrt(cp.mean(x**2) - x_mean**2)

    # Gi* = (sum_j(w_ij * x_j) - x_mean * sum_j(w_ij)) /
    #        (S * sqrt((n * sum_j(w_ij^2) - (sum_j(w_ij))^2) / (n-1)))
    Wx = W @ x  # weighted sum
    Wi_sum = W.sum(axis=1)
    Wi2_sum = (W**2).sum(axis=1)

    numerator = Wx - x_mean * Wi_sum
    denominator = S * cp.sqrt((n_f * Wi2_sum - Wi_sum**2) / (n_f - 1))
    denominator = cp.where(denominator == 0, 1.0, denominator)

    gi_star_z = cp.asnumpy(numerator / denominator)
    # Convert z-scores to p-values (two-tailed)
    gi_star_p = 2 * (1 - scipy_stats.norm.cdf(np.abs(gi_star_z)))
    method = "GPU (CuPy)"
elif PYSAL:
    w_queen = Queen.from_dataframe(zones_pu)
    w_queen.transform = "R"
    gi = G_Local(trip_counts, w_queen, star=True, permutations=0)
    gi_star_z = gi.Zs
    gi_star_p = gi.p_norm
    method = "PySAL"
else:
    gi_star_z = np.zeros(n)
    gi_star_p = np.ones(n)
    method = "Skipped"

# Classify into ArcGIS-style confidence bins
def classify_hotspot(z, p):
    if p < 0.01 and z > 0: return "Hot Spot (99%)"
    elif p < 0.05 and z > 0: return "Hot Spot (95%)"
    elif p < 0.10 and z > 0: return "Hot Spot (90%)"
    elif p < 0.01 and z < 0: return "Cold Spot (99%)"
    elif p < 0.05 and z < 0: return "Cold Spot (95%)"
    elif p < 0.10 and z < 0: return "Cold Spot (90%)"
    else: return "Not Significant"

zones_hotspot = zones.copy()
zones_hotspot["gi_z_score"] = np.round(gi_star_z, 4)
zones_hotspot["gi_p_value"] = np.round(gi_star_p, 6)
zones_hotspot["hotspot_class"] = [classify_hotspot(z, p) for z, p in zip(gi_star_z, gi_star_p)]
zones_hotspot["trip_count"] = trip_counts.astype(int)
zones_hotspot["geometry"] = zones_hotspot["geometry"].simplify(0.0005, preserve_topology=True)

save_geojson(zones_hotspot, "hotspot_analysis.geojson")

hot = sum(1 for c in zones_hotspot["hotspot_class"] if "Hot" in c)
cold = sum(1 for c in zones_hotspot["hotspot_class"] if "Cold" in c)
print(f"  Gi* computed via {method}: {hot} hot spots, {cold} cold spots ({time.time()-t0:.2f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GPU-ACCELERATED MORAN'S I SPATIAL AUTOCORRELATION
#    (Equivalent to ArcGIS Pro "Spatial Autocorrelation (Global Moran's I)")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GPU-ACCELERATED MORAN'S I SPATIAL AUTOCORRELATION")
print(f"{'='*70}")
t0 = time.time()

if GPU:
    x = cp.asarray(trip_counts, dtype=cp.float64)
    W = cp.asarray(w_matrix, dtype=cp.float64)

    x_dev = x - cp.mean(x)
    n_f = cp.float64(n)
    S0 = W.sum()

    numerator = n_f * float(cp.sum(W * cp.outer(x_dev, x_dev)))
    denominator = S0 * float(cp.sum(x_dev**2))

    morans_I = float(numerator / denominator) if denominator != 0 else 0.0

    # Expected I under null
    E_I = -1.0 / (n - 1)
    # Variance (normality assumption)
    S1 = float(cp.sum((W + W.T)**2)) / 2
    S2 = float(cp.sum((W.sum(axis=0) + W.sum(axis=1))**2))
    k = float(n_f * cp.sum(x_dev**4) / (cp.sum(x_dev**2)**2))

    A = n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2)
    B = k * ((n**2 - n)*S1 - 2*n*S2 + 6*S0**2)
    C = (n-1)*(n-2)*(n-3)*S0**2
    V_I = (A - B) / C - E_I**2

    z_I = float((morans_I - E_I) / np.sqrt(V_I)) if V_I > 0 else 0.0
    p_I = float(2 * (1 - scipy_stats.norm.cdf(abs(z_I))))
    method = "GPU (CuPy)"
elif PYSAL:
    w_queen = Queen.from_dataframe(zones)
    w_queen.transform = "R"
    mi = Moran(trip_counts, w_queen)
    morans_I = mi.I
    z_I = mi.z_norm
    p_I = mi.p_norm
    method = "PySAL"
else:
    morans_I = z_I = p_I = 0
    method = "Skipped"

spatial_autocorr = {
    "morans_I": round(morans_I, 6),
    "expected_I": round(-1/(n-1), 6),
    "z_score": round(z_I, 4),
    "p_value": round(p_I, 6),
    "interpretation": "Clustered" if morans_I > 0 and p_I < 0.05
                      else "Dispersed" if morans_I < 0 and p_I < 0.05
                      else "Random",
    "method": method,
}
print(f"  Moran's I = {morans_I:.4f} (z={z_I:.2f}, p={p_I:.6f})")
print(f"  Pattern: {spatial_autocorr['interpretation']} — via {method} ({time.time()-t0:.2f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GPU-ACCELERATED KERNEL DENSITY ESTIMATION
#    (Equivalent to ArcGIS Pro "Kernel Density" tool)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GPU-ACCELERATED KERNEL DENSITY ESTIMATION")
print(f"{'='*70}")
t0 = time.time()

# Weighted centroids: repeat by trip count (capped for memory)
active = zones_pu[zones_pu["trip_count"] > 0]
weights = np.minimum(active["trip_count"].astype(int).values, 1000)
sample_x = np.repeat(active["centroid_x"].values, weights)
sample_y = np.repeat(active["centroid_y"].values, weights)

xmin, ymin, xmax, ymax = zones.total_bounds
pad = 0.005
grid_size = 200
xx = np.linspace(xmin - pad, xmax + pad, grid_size)
yy = np.linspace(ymin - pad, ymax + pad, grid_size)

if GPU:
    gx = cp.asarray(np.tile(xx, grid_size))
    gy = cp.asarray(np.repeat(yy, grid_size))
    sx = cp.asarray(sample_x)
    sy = cp.asarray(sample_y)

    bandwidth = 0.008
    density = cp.zeros(grid_size * grid_size, dtype=cp.float64)

    batch = 5000
    for i in range(0, len(sx), batch):
        bx = sx[i:i+batch]
        by = sy[i:i+batch]
        d2 = (gx[:, None] - bx[None, :])**2 + (gy[:, None] - by[None, :])**2
        density += cp.sum(cp.exp(-d2 / (2 * bandwidth**2)), axis=1)

    density_np = cp.asnumpy(density).reshape(grid_size, grid_size)
    method = "GPU (CuPy)"
else:
    kernel = scipy_stats.gaussian_kde(np.vstack([sample_x, sample_y]), bw_method=0.05)
    grid_points = np.vstack([np.tile(xx, grid_size), np.repeat(yy, grid_size)])
    density_np = kernel(grid_points).reshape(grid_size, grid_size)
    method = "CPU (SciPy)"

density_np = (density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-10)

heatmap_features = []
for i in range(grid_size):
    for j in range(grid_size):
        if density_np[i, j] > 0.03:
            heatmap_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(xx[j]), float(yy[i])]},
                "properties": {"density": round(float(density_np[i, j]), 4)}
            })

save_geojson({"type": "FeatureCollection", "features": heatmap_features}, "kde_heatmap.geojson")
print(f"  KDE via {method}: {len(heatmap_features)} grid points ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GPU-ACCELERATED IDW INTERPOLATION
#    (Equivalent to ArcGIS Pro "IDW" interpolation tool)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  GPU-ACCELERATED IDW INTERPOLATION (Fare Surface)")
print(f"{'='*70}")
t0 = time.time()

# Interpolate average fare across a grid using inverse distance weighting
known_x = active["centroid_x"].values
known_y = active["centroid_y"].values
known_vals = active["avg_fare"].values

idw_size = 100
ix = np.linspace(xmin, xmax, idw_size)
iy = np.linspace(ymin, ymax, idw_size)

if GPU:
    gx = cp.asarray(np.tile(ix, idw_size))
    gy = cp.asarray(np.repeat(iy, idw_size))
    kx = cp.asarray(known_x)
    ky = cp.asarray(known_y)
    kv = cp.asarray(known_vals)

    dx = gx[:, None] - kx[None, :]
    dy = gy[:, None] - ky[None, :]
    dist = cp.sqrt(dx**2 + dy**2)
    dist = cp.maximum(dist, 1e-10)

    power = 2.0
    weights_gpu = 1.0 / dist**power
    idw_vals = cp.sum(weights_gpu * kv[None, :], axis=1) / cp.sum(weights_gpu, axis=1)
    idw_np = cp.asnumpy(idw_vals).reshape(idw_size, idw_size)
    method = "GPU (CuPy)"
else:
    gx = np.tile(ix, idw_size)
    gy = np.repeat(iy, idw_size)
    dx = gx[:, None] - known_x[None, :]
    dy = gy[:, None] - known_y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1e-10)
    weights_np = 1.0 / dist**2
    idw_vals = np.sum(weights_np * known_vals[None, :], axis=1) / np.sum(weights_np, axis=1)
    idw_np = idw_vals.reshape(idw_size, idw_size)
    method = "CPU (NumPy)"

# Export as GeoJSON points
idw_features = []
fare_min, fare_max = float(idw_np.min()), float(idw_np.max())
for i in range(idw_size):
    for j in range(idw_size):
        val = float(idw_np[i, j])
        if fare_min < val:
            idw_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(ix[j]), float(iy[i])]},
                "properties": {"fare": round(val, 2)}
            })

save_geojson({"type": "FeatureCollection", "features": idw_features}, "idw_fare_surface.geojson")
print(f"  IDW via {method}: {len(idw_features)} points, range ${fare_min:.2f}-${fare_max:.2f} ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. H3 HEXAGONAL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  H3 HEXAGONAL SPATIAL BINNING")
print(f"{'='*70}")
t0 = time.time()

active2 = zones_pu[zones_pu["trip_count"] > 0].copy()
active2["h3_index"] = active2.apply(
    lambda r: h3.latlng_to_cell(r["centroid_y"], r["centroid_x"], 8), axis=1
)

h3_stats = active2.groupby("h3_index").agg(
    trip_count=("trip_count", "sum"),
    avg_fare=("avg_fare", "mean"),
    avg_tip=("avg_tip", "mean"),
    total_revenue=("total_revenue", "sum"),
).reset_index()
for c in ["avg_fare", "avg_tip", "total_revenue"]:
    h3_stats[c] = h3_stats[c].round(2)

h3_features = []
for _, row in h3_stats.iterrows():
    boundary = h3.cell_to_boundary(row["h3_index"])
    coords = [[lng, lat] for lat, lng in boundary]
    coords.append(coords[0])
    h3_features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {
            "h3_index": row["h3_index"],
            "trip_count": int(row["trip_count"]),
            "avg_fare": row["avg_fare"],
            "avg_tip": row["avg_tip"],
            "total_revenue": row["total_revenue"],
        }
    })

save_geojson({"type": "FeatureCollection", "features": h3_features}, "h3_hexagons.geojson")
print(f"  {len(h3_features)} hexagons at resolution 8 ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. DBSCAN SPATIAL CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  DBSCAN SPATIAL CLUSTERING")
print(f"{'='*70}")
t0 = time.time()

cluster_data = zones_pu[zones_pu["trip_count"] > 500].copy()
coords_arr = np.column_stack([cluster_data["centroid_x"].values, cluster_data["centroid_y"].values])

clustering = DBSCAN(eps=0.008, min_samples=3).fit(
    coords_arr, sample_weight=np.log1p(cluster_data["trip_count"].values)
)
cluster_data = cluster_data.copy()
cluster_data["cluster"] = clustering.labels_

cluster_gdf = gpd.GeoDataFrame(
    cluster_data[["LocationID", "zone", "borough", "trip_count", "avg_fare",
                   "total_revenue", "cluster"]],
    geometry=gpd.points_from_xy(cluster_data["centroid_x"], cluster_data["centroid_y"]),
    crs="EPSG:4326"
)
save_geojson(cluster_gdf, "dbscan_clusters.geojson")
n_clusters = len(set(clustering.labels_) - {-1})
print(f"  {n_clusters} clusters from {len(cluster_data)} zones ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. OD FLOW ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  ORIGIN-DESTINATION FLOW ANALYSIS")
print(f"{'='*70}")
t0 = time.time()

od = trips.groupby(["PULocationID", "DOLocationID"]).agg(
    flow_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_duration=("duration_min", "mean"),
    avg_distance=("trip_distance", "mean"),
).reset_index()
od["avg_fare"] = od["avg_fare"].round(2)
od["avg_duration"] = od["avg_duration"].round(1)
od["avg_distance"] = od["avg_distance"].round(2)

top_od = od.nlargest(200, "flow_count")

centroid_lut = {r["LocationID"]: (r["centroid_x"], r["centroid_y"]) for _, r in zones.iterrows()}
zone_lut = {r["LocationID"]: r["zone"] for _, r in zones.iterrows()}

od_features = []
for _, row in top_od.iterrows():
    pu = centroid_lut.get(row["PULocationID"])
    do = centroid_lut.get(row["DOLocationID"])
    if pu and do:
        od_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [list(pu), list(do)]},
            "properties": {
                "from": zone_lut.get(row["PULocationID"], ""),
                "to": zone_lut.get(row["DOLocationID"], ""),
                "flow_count": int(row["flow_count"]),
                "avg_fare": row["avg_fare"],
                "avg_duration": row["avg_duration"],
                "avg_distance": row["avg_distance"],
            }
        })

save_geojson({"type": "FeatureCollection", "features": od_features}, "od_flows.geojson")
print(f"  Top {len(od_features)} OD flows ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  TEMPORAL ANALYSIS")
print(f"{'='*70}")
t0 = time.time()

temporal = trips.groupby(["pickup_dow", "pickup_hour"]).agg(
    trip_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_tip=("tip_amount", "mean"),
    avg_distance=("trip_distance", "mean"),
).reset_index()
temporal["avg_fare"] = temporal["avg_fare"].round(2)
temporal["avg_tip"] = temporal["avg_tip"].round(2)
temporal["avg_distance"] = temporal["avg_distance"].round(2)

save_json(temporal.to_dict(orient="records"), "temporal_patterns.json")
print(f"  {len(temporal)} time bins ({time.time()-t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. ZONE BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════════════

zones_simple = zones[["LocationID", "zone", "borough", "geometry"]].copy()
zones_simple["geometry"] = zones_simple["geometry"].simplify(0.0005, preserve_topology=True)
save_geojson(zones_simple, "taxi_zones.geojson")

# ═══════════════════════════════════════════════════════════════════════════════
# 13. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

summary = {
    "total_trips": int(len(trips)),
    "total_revenue": round(float(trips["total_amount"].sum()), 2),
    "avg_fare": round(float(trips["fare_amount"].mean()), 2),
    "avg_tip": round(float(trips["tip_amount"].mean()), 2),
    "avg_distance_miles": round(float(trips["trip_distance"].mean()), 2),
    "avg_duration_min": round(float(trips["duration_min"].mean()), 1),
    "median_fare": round(float(trips["fare_amount"].median()), 2),
    "busiest_zone": zones_pu.loc[zones_pu["trip_count"].idxmax(), "zone"],
    "busiest_zone_trips": int(zones_pu["trip_count"].max()),
    "total_zones": n,
    "dbscan_clusters": n_clusters,
    "h3_hexagons": len(h3_features),
    "hot_spots": hot,
    "cold_spots": cold,
    "morans_I": spatial_autocorr,
    "gpu_accelerated": GPU,
    "data_period": "January 2024",
    "data_source": "NYC Taxi & Limousine Commission",
    "analyses": [
        "Zone Aggregation (Pickup/Dropoff/Revenue/Tip)",
        "GPU Getis-Ord Gi* Hot Spot Analysis",
        "GPU Moran's I Spatial Autocorrelation",
        "GPU Kernel Density Estimation",
        "GPU IDW Interpolation (Fare Surface)",
        "GPU Pairwise Distance Matrix",
        "H3 Hexagonal Aggregation",
        "DBSCAN Spatial Clustering",
        "Origin-Destination Flow Analysis",
        "Temporal Pattern Analysis",
    ]
}
save_json(summary, "summary.json")

print(f"\n{'='*70}")
print("  COMPLETE")
print(f"{'='*70}")
for k, v in summary.items():
    if k != "analyses" and k != "morans_I":
        print(f"  {k}: {v}")
print(f"\n  Outputs: {OUTPUT}")
print(f"  Web data: {WEBAPP_DATA}")
