# GPU-Accelerated Spatial Analytics — NYC Taxi

Interactive spatial analytics platform combining **NVIDIA GPU acceleration** (CuPy) with the **ArcGIS Maps SDK for JavaScript 5.0** to analyze 2.8 million NYC Yellow Taxi trips from January 2024.

Implements ArcGIS Pro-equivalent spatial analysis tools (Hot Spot Analysis, Kernel Density, IDW, Spatial Autocorrelation) accelerated on the GPU via CuPy matrix operations.

![NVIDIA RTX Accelerated](https://img.shields.io/badge/NVIDIA-RTX%20Accelerated-76b900?style=flat-square&logo=nvidia)
![ArcGIS Maps SDK](https://img.shields.io/badge/ArcGIS%20Maps%20SDK-5.0-0079c1?style=flat-square&logo=esri)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

## Live Demo

**[View the Interactive Map](https://garridolecca.github.io/Nvidia_esri-gpu-spatial-analytics/)**

## What This Project Does

GPU-accelerated implementations of standard GIS spatial analysis tools, with results visualized through ArcGIS Maps SDK 5.0 web components.

### GPU-Accelerated Analytics (ArcGIS Equivalents)

| Analysis | ArcGIS Pro Equivalent | GPU Method |
|---|---|---|
| **Getis-Ord Gi* Hot Spot Analysis** | Hot Spot Analysis (Gi*) | CuPy matrix multiply on spatial weights |
| **Global Moran's I** | Spatial Autocorrelation | CuPy outer product + weighted sums |
| **Kernel Density Estimation** | Kernel Density tool | CuPy batched pairwise Gaussian kernel |
| **IDW Interpolation** | IDW tool | CuPy full distance matrix + weighted average |
| **Pairwise Distance Matrix** | Generate Near Table | CuPy Haversine-approximation |
| **Spatial Weights Matrix** | Generate Spatial Weights | CuPy inverse-distance with threshold |

### Additional Analytics

| Analysis | Method |
|---|---|
| Zone Aggregation (Pickup/Dropoff/Revenue/Tip) | Pandas groupby |
| H3 Hexagonal Spatial Binning | Uber H3 resolution 8 |
| DBSCAN Spatial Clustering | scikit-learn |
| Origin-Destination Flow Analysis | Top 200 OD pairs |
| Temporal Pattern Analysis | 168 hour-of-week bins |

### Web Visualization (ArcGIS Maps SDK 5.0)

Built with the new **web component architecture** (`<arcgis-map>`, `$arcgis.import()`) — 12 interactive map layers:

- **Hot Spot Analysis** — Getis-Ord Gi* classification (99%/95%/90% confidence)
- **Kernel Density** — GPU-computed continuous density surface
- **IDW Fare Surface** — Interpolated average fare across NYC
- **Spatial Autocorrelation** — Moran's I statistics panel
- **Pickup/Dropoff Volume** — Zone-level choropleth maps
- **H3 Hexagons** — Uniform hexagonal aggregation
- **DBSCAN Clusters** — 53 spatial clusters
- **OD Flows** — Top 200 busiest routes
- **Temporal Patterns** — Interactive hour-of-week heatmap
- **Revenue & Tips** — Financial analysis choropleths

## Key Findings

| Metric | Value |
|---|---|
| **Moran's I** | 0.5222 (z=23.38, p<0.0001) — Strong spatial clustering |
| **Hot Spots** | 59 zones (Midtown, Times Square, Financial District) |
| **Cold Spots** | 70 zones (outer boroughs, residential areas) |
| **Busiest Zone** | Midtown Center — 139,757 pickups |
| **Total Revenue** | $76.7M across all trips |
| **Average Trip** | 3.23 mi, 15.7 min, $18.16 fare, $3.37 tip |
| **DBSCAN Clusters** | 53 clusters, concentrated in Manhattan |

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| NYC Yellow Taxi Trips (Jan 2024) | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ~2.96M rows |
| NYC Taxi Zone Boundaries | [NYC TLC](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip) | 263 zones |

## Project Structure

```
gpu-spatial-analytics/
├── scripts/
│   ├── download_data.py          # Downloads taxi data + zone boundaries
│   └── run_analytics.py          # GPU-accelerated analytics pipeline (10 analyses)
├── webapp/
│   ├── index.html                # ArcGIS Maps SDK 5.0 web app (12 layers)
│   └── data/                     # Pre-computed analytics output
│       ├── hotspot_analysis.geojson   # Getis-Ord Gi* results
│       ├── kde_heatmap.geojson        # Kernel density surface
│       ├── idw_fare_surface.geojson   # IDW interpolated fares
│       ├── zone_pickups.geojson       # Pickup aggregation
│       ├── zone_dropoffs.geojson      # Dropoff aggregation
│       ├── h3_hexagons.geojson        # H3 hex aggregation
│       ├── dbscan_clusters.geojson    # DBSCAN cluster assignments
│       ├── od_flows.geojson           # Top 200 OD flows
│       ├── temporal_patterns.json     # Hour-of-week patterns
│       └── summary.json              # Dataset summary + Moran's I
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (for GPU acceleration; falls back to CPU)
- ~50 MB disk space for data

### Run the Analytics

```bash
pip install -r requirements.txt
python scripts/download_data.py     # Download data (~48 MB)
python scripts/run_analytics.py     # Run GPU analytics pipeline
```

### View the Web App

```bash
cd webapp && python -m http.server 8000
# Open http://localhost:8000
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| GPU Compute | **NVIDIA CuPy** | Matrix operations, KDE, IDW, Gi*, Moran's I |
| Spatial Data | **GeoPandas + Shapely** | GeoJSON I/O, spatial operations |
| Spatial Stats | **PySAL (libpysal + esda)** | Reference implementations for validation |
| Hex Indexing | **Uber H3** | Hexagonal spatial binning |
| Clustering | **scikit-learn** | DBSCAN density-based clustering |
| Web Mapping | **ArcGIS Maps SDK 5.0** | Web components, GeoJSONLayer, renderers |
| Basemaps | **Esri Dark Gray Vector** | Dark-themed cartographic base |

## How GPU Acceleration Works

The pipeline uses CuPy to offload computationally intensive spatial operations to the NVIDIA GPU:

1. **Spatial Weights Matrix** — Constructs a 263x263 inverse-distance weight matrix on the GPU with a 5km threshold. Row-standardized for use in spatial statistics.

2. **Getis-Ord Gi\*** — Computes the full Gi* statistic via GPU matrix multiplication: `W @ x` for the weighted sum, then vectorized z-score computation across all zones simultaneously.

3. **Moran's I** — Computes the global spatial autocorrelation index using GPU outer products and elementwise operations on the spatial weights matrix.

4. **Kernel Density** — Evaluates a Gaussian kernel (bandwidth ~800m) across a 200x200 grid. Batched pairwise distance computation between 40,000 grid cells and thousands of weighted sample points.

5. **IDW Interpolation** — Full distance matrix between 10,000 grid points and all zone centroids, with power-2 inverse distance weighting computed in a single GPU operation.

When no GPU is available, the pipeline automatically falls back to NumPy/SciPy implementations.

## License

MIT License. Data provided by NYC Taxi & Limousine Commission under [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse).
