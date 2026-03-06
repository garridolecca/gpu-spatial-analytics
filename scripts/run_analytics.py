"""
GPU-Accelerated Spatial Analytics Pipeline
Uses CuPy for GPU-accelerated computations on NYC Yellow Taxi data (Jan 2024).
Outputs GeoJSON files for visualization in ArcGIS JavaScript SDK.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from scipy import stats
from sklearn.cluster import DBSCAN

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled (CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available, falling back to NumPy")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("\n=== Loading Data ===")
t0 = time.time()

zones = gpd.read_file(os.path.join(DATA, "taxi_zones", "taxi_zones", "taxi_zones.shp"))
zones = zones.to_crs(epsg=4326)  # reproject to WGS84 for web mapping
print(f"  Loaded {len(zones)} taxi zones")

trips = pd.read_parquet(os.path.join(DATA, "yellow_tripdata_2024-01.parquet"))
print(f"  Loaded {len(trips):,} taxi trips ({time.time()-t0:.1f}s)")

# Clean: remove trips with invalid zone IDs or amounts
valid_ids = set(zones["LocationID"].values)
trips = trips[
    trips["PULocationID"].isin(valid_ids) &
    trips["DOLocationID"].isin(valid_ids) &
    (trips["total_amount"] > 0) &
    (trips["trip_distance"] > 0) &
    (trips["trip_distance"] < 100)
].copy()
print(f"  After cleaning: {len(trips):,} trips")

# ── 2. Zone-level pickup aggregation ────────────────────────────────────────

print("\n=== Zone-Level Pickup Analytics ===")
t0 = time.time()

pu_stats = trips.groupby("PULocationID").agg(
    trip_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_tip=("tip_amount", "mean"),
    avg_distance=("trip_distance", "mean"),
    avg_total=("total_amount", "mean"),
    total_revenue=("total_amount", "sum"),
    avg_passengers=("passenger_count", "mean"),
).reset_index()
pu_stats.columns = ["LocationID", "trip_count", "avg_fare", "avg_tip",
                     "avg_distance", "avg_total", "total_revenue", "avg_passengers"]

# Round floats
for col in ["avg_fare", "avg_tip", "avg_distance", "avg_total", "total_revenue", "avg_passengers"]:
    pu_stats[col] = pu_stats[col].round(2)

zones_pu = zones.merge(pu_stats, on="LocationID", how="left").fillna(0)
zones_pu.to_file(os.path.join(OUTPUT, "zone_pickups.geojson"), driver="GeoJSON")
print(f"  Zone pickup stats computed ({time.time()-t0:.1f}s)")

# ── 3. Zone-level dropoff aggregation ───────────────────────────────────────

print("\n=== Zone-Level Dropoff Analytics ===")
t0 = time.time()

do_stats = trips.groupby("DOLocationID").agg(
    dropoff_count=("VendorID", "count"),
    avg_fare_do=("fare_amount", "mean"),
).reset_index()
do_stats.columns = ["LocationID", "dropoff_count", "avg_fare_do"]
do_stats["avg_fare_do"] = do_stats["avg_fare_do"].round(2)

zones_do = zones.merge(do_stats, on="LocationID", how="left").fillna(0)
zones_do.to_file(os.path.join(OUTPUT, "zone_dropoffs.geojson"), driver="GeoJSON")
print(f"  Zone dropoff stats computed ({time.time()-t0:.1f}s)")

# ── 4. GPU-Accelerated Kernel Density Estimation (Hotspot Heatmap) ─────────

print("\n=== GPU-Accelerated KDE Hotspot Analysis ===")
t0 = time.time()

# Get zone centroids as pickup locations (weighted by trip count)
centroids = zones_pu[zones_pu["trip_count"] > 0].copy()
centroids["cx"] = centroids.geometry.centroid.x
centroids["cy"] = centroids.geometry.centroid.y

# Generate a dense grid over NYC for KDE
xmin, ymin, xmax, ymax = zones.total_bounds
grid_size = 200
xx = np.linspace(xmin, xmax, grid_size)
yy = np.linspace(ymin, ymax, grid_size)
xx_grid, yy_grid = np.meshgrid(xx, yy)
grid_points = np.vstack([xx_grid.ravel(), yy_grid.ravel()])  # 2 x N

# Weighted sample points (repeat centroids by trip count, capped)
sample_x = np.repeat(centroids["cx"].values, np.minimum(centroids["trip_count"].astype(int).values, 1000))
sample_y = np.repeat(centroids["cy"].values, np.minimum(centroids["trip_count"].astype(int).values, 1000))

if GPU_AVAILABLE:
    # GPU-accelerated pairwise distance + Gaussian KDE
    gx = cp.asarray(grid_points[0])  # grid x coords
    gy = cp.asarray(grid_points[1])  # grid y coords
    sx = cp.asarray(sample_x)
    sy = cp.asarray(sample_y)

    bandwidth = 0.008  # ~800m in degrees at NYC latitude
    density = cp.zeros(len(gx), dtype=cp.float64)

    # Process in batches to fit GPU memory
    batch_size = 5000
    for i in range(0, len(sx), batch_size):
        bx = sx[i:i+batch_size]
        by = sy[i:i+batch_size]
        dx = gx[:, None] - bx[None, :]
        dy = gy[:, None] - by[None, :]
        dist_sq = dx**2 + dy**2
        density += cp.sum(cp.exp(-dist_sq / (2 * bandwidth**2)), axis=1)

    density_np = cp.asnumpy(density).reshape(grid_size, grid_size)
else:
    kernel = stats.gaussian_kde(np.vstack([sample_x, sample_y]), bw_method=0.05)
    density_np = kernel(grid_points).reshape(grid_size, grid_size)

# Normalize to 0-1
density_np = (density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-10)

# Convert to GeoJSON point grid (only cells with density > 0.05 to reduce file size)
heatmap_features = []
for i in range(grid_size):
    for j in range(grid_size):
        if density_np[i, j] > 0.05:
            heatmap_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(xx[j]), float(yy[i])]},
                "properties": {"density": round(float(density_np[i, j]), 4)}
            })

heatmap_geojson = {"type": "FeatureCollection", "features": heatmap_features}
with open(os.path.join(OUTPUT, "kde_heatmap.geojson"), "w") as f:
    json.dump(heatmap_geojson, f)
print(f"  KDE heatmap: {len(heatmap_features)} grid points ({time.time()-t0:.1f}s)")

# ── 5. H3 Hexagonal Aggregation ────────────────────────────────────────────

print("\n=== H3 Hexagonal Aggregation ===")
t0 = time.time()

# Assign each zone centroid an H3 hex, aggregate trips
centroids["h3_index"] = centroids.apply(
    lambda r: h3.latlng_to_cell(r["cy"], r["cx"], 8), axis=1
)

h3_stats = centroids.groupby("h3_index").agg(
    trip_count=("trip_count", "sum"),
    avg_fare=("avg_fare", "mean"),
    avg_tip=("avg_tip", "mean"),
    total_revenue=("total_revenue", "sum"),
).reset_index()
h3_stats["avg_fare"] = h3_stats["avg_fare"].round(2)
h3_stats["avg_tip"] = h3_stats["avg_tip"].round(2)
h3_stats["total_revenue"] = h3_stats["total_revenue"].round(2)

# Convert H3 hexes to GeoJSON polygons
h3_features = []
for _, row in h3_stats.iterrows():
    boundary = h3.cell_to_boundary(row["h3_index"])
    # h3 returns (lat, lng) pairs, GeoJSON needs (lng, lat)
    coords = [[lng, lat] for lat, lng in boundary]
    coords.append(coords[0])  # close ring
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

h3_geojson = {"type": "FeatureCollection", "features": h3_features}
with open(os.path.join(OUTPUT, "h3_hexagons.geojson"), "w") as f:
    json.dump(h3_geojson, f)
print(f"  H3 hexagons: {len(h3_features)} hexes ({time.time()-t0:.1f}s)")

# ── 6. DBSCAN Spatial Clustering ───────────────────────────────────────────

print("\n=== DBSCAN Spatial Clustering ===")
t0 = time.time()

cluster_data = centroids[centroids["trip_count"] > 500].copy()
coords_arr = np.column_stack([cluster_data["cx"].values, cluster_data["cy"].values])

# eps in degrees (~500m at NYC latitude)
clustering = DBSCAN(eps=0.008, min_samples=3).fit(
    coords_arr, sample_weight=np.log1p(cluster_data["trip_count"].values)
)
cluster_data["cluster"] = clustering.labels_

# Build cluster summary
cluster_gdf = gpd.GeoDataFrame(
    cluster_data[["LocationID", "zone", "borough", "trip_count", "avg_fare",
                   "total_revenue", "cluster", "cx", "cy"]],
    geometry=gpd.points_from_xy(cluster_data["cx"], cluster_data["cy"]),
    crs="EPSG:4326"
)
cluster_gdf.to_file(os.path.join(OUTPUT, "dbscan_clusters.geojson"), driver="GeoJSON")
n_clusters = len(set(clustering.labels_) - {-1})
print(f"  Found {n_clusters} clusters from {len(cluster_data)} zones ({time.time()-t0:.1f}s)")

# ── 7. GPU-Accelerated OD Flow Matrix ─────────────────────────────────────

print("\n=== GPU-Accelerated OD Flow Analysis ===")
t0 = time.time()

od_counts = trips.groupby(["PULocationID", "DOLocationID"]).agg(
    flow_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_duration_min=("tpep_dropoff_datetime", lambda x: None),  # placeholder
).reset_index()

# Compute average trip duration
trips["duration_min"] = (trips["tpep_dropoff_datetime"] - trips["tpep_pickup_datetime"]).dt.total_seconds() / 60
od_dur = trips.groupby(["PULocationID", "DOLocationID"])["duration_min"].mean().reset_index()
od_counts = od_counts.drop(columns=["avg_duration_min"]).merge(od_dur, on=["PULocationID", "DOLocationID"])
od_counts["avg_fare"] = od_counts["avg_fare"].round(2)
od_counts["duration_min"] = od_counts["duration_min"].round(1)

# Top 200 OD flows for visualization
top_od = od_counts.nlargest(200, "flow_count")

# Build centroid lookup
centroid_lookup = {}
for _, row in zones.iterrows():
    c = row.geometry.centroid
    centroid_lookup[row["LocationID"]] = (c.x, c.y)

od_features = []
for _, row in top_od.iterrows():
    pu = centroid_lookup.get(row["PULocationID"])
    do = centroid_lookup.get(row["DOLocationID"])
    if pu and do:
        od_features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(pu), list(do)]
            },
            "properties": {
                "pu_zone": int(row["PULocationID"]),
                "do_zone": int(row["DOLocationID"]),
                "flow_count": int(row["flow_count"]),
                "avg_fare": row["avg_fare"],
                "duration_min": row["duration_min"],
            }
        })

od_geojson = {"type": "FeatureCollection", "features": od_features}
with open(os.path.join(OUTPUT, "od_flows.geojson"), "w") as f:
    json.dump(od_geojson, f)
print(f"  Top {len(od_features)} OD flows exported ({time.time()-t0:.1f}s)")

# ── 8. GPU-Accelerated Distance Matrix Between Zone Centroids ──────────────

print("\n=== GPU-Accelerated Pairwise Distance Matrix ===")
t0 = time.time()

all_cx = zones.geometry.centroid.x.values
all_cy = zones.geometry.centroid.y.values

if GPU_AVAILABLE:
    cx_gpu = cp.asarray(all_cx)
    cy_gpu = cp.asarray(all_cy)
    # Haversine-approximation using GPU (degrees to km at NYC latitude)
    dx = (cx_gpu[:, None] - cx_gpu[None, :]) * 85.0  # ~85 km per degree longitude at 40.7N
    dy = (cy_gpu[:, None] - cy_gpu[None, :]) * 111.0  # ~111 km per degree latitude
    dist_matrix = cp.sqrt(dx**2 + dy**2)
    dist_np = cp.asnumpy(dist_matrix)
    method = "GPU (CuPy)"
else:
    dx = (all_cx[:, None] - all_cx[None, :]) * 85.0
    dy = (all_cy[:, None] - all_cy[None, :]) * 111.0
    dist_np = np.sqrt(dx**2 + dy**2)
    method = "CPU (NumPy)"

print(f"  {len(zones)}x{len(zones)} distance matrix computed via {method}")
print(f"  Mean inter-zone distance: {dist_np.mean():.2f} km")
print(f"  Max inter-zone distance: {dist_np.max():.2f} km ({time.time()-t0:.2f}s)")

# ── 9. Temporal Analysis ───────────────────────────────────────────────────

print("\n=== Temporal Analysis ===")
t0 = time.time()

trips["pickup_hour"] = trips["tpep_pickup_datetime"].dt.hour
trips["pickup_dow"] = trips["tpep_pickup_datetime"].dt.dayofweek  # 0=Monday

temporal = trips.groupby(["pickup_dow", "pickup_hour"]).agg(
    trip_count=("VendorID", "count"),
    avg_fare=("fare_amount", "mean"),
    avg_tip_pct=("tip_amount", lambda x: (x / trips.loc[x.index, "fare_amount"].replace(0, np.nan)).mean() * 100),
).reset_index()
temporal["avg_fare"] = temporal["avg_fare"].round(2)
temporal["avg_tip_pct"] = temporal["avg_tip_pct"].round(1)

temporal_data = temporal.to_dict(orient="records")
with open(os.path.join(OUTPUT, "temporal_patterns.json"), "w") as f:
    json.dump(temporal_data, f)
print(f"  Temporal patterns: {len(temporal_data)} hour-dow combos ({time.time()-t0:.1f}s)")

# ── 10. Summary statistics ─────────────────────────────────────────────────

print("\n=== Summary Statistics ===")
summary = {
    "total_trips": int(len(trips)),
    "total_revenue": round(float(trips["total_amount"].sum()), 2),
    "avg_fare": round(float(trips["fare_amount"].mean()), 2),
    "avg_tip": round(float(trips["tip_amount"].mean()), 2),
    "avg_distance_miles": round(float(trips["trip_distance"].mean()), 2),
    "avg_duration_min": round(float(trips["duration_min"].mean()), 1),
    "busiest_zone": zones_pu.loc[zones_pu["trip_count"].idxmax(), "zone"],
    "busiest_zone_trips": int(zones_pu["trip_count"].max()),
    "total_zones": int(len(zones)),
    "dbscan_clusters": n_clusters,
    "h3_hexagons": len(h3_features),
    "gpu_accelerated": GPU_AVAILABLE,
    "data_period": "January 2024",
    "data_source": "NYC Taxi & Limousine Commission",
}
with open(os.path.join(OUTPUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

for k, v in summary.items():
    print(f"  {k}: {v}")

# Export zone boundaries as simplified GeoJSON for the web app
zones_simple = zones[["LocationID", "zone", "borough", "geometry"]].copy()
zones_simple.to_file(os.path.join(OUTPUT, "taxi_zones.geojson"), driver="GeoJSON")

print(f"\nAll outputs saved to {OUTPUT}")
print("Pipeline complete!")
