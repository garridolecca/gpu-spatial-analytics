# GPU-Accelerated Spatial Analytics — NYC Taxi

Interactive spatial analytics platform combining **NVIDIA GPU acceleration** (CuPy) with the **ArcGIS JavaScript SDK** to analyze 2.8 million NYC Yellow Taxi trips from January 2024.

![NVIDIA RTX Accelerated](https://img.shields.io/badge/NVIDIA-RTX%20Accelerated-76b900?style=flat-square&logo=nvidia)
![ArcGIS JS SDK](https://img.shields.io/badge/ArcGIS%20JS%20SDK-4.30-0079c1?style=flat-square&logo=esri)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

## Live Demo

**[View the Interactive Map](https://PLACEHOLDER)**

## What This Project Does

This project demonstrates how GPU computing can accelerate geospatial analytics at scale, then visualizes the results through an interactive web application built with Esri's ArcGIS JavaScript SDK.

### Analytics Pipeline (Python + CuPy GPU)

| Analysis | Method | GPU-Accelerated |
|---|---|:---:|
| **Zone Pickup/Dropoff Aggregation** | Pandas groupby on 2.8M trips across 263 zones | — |
| **Kernel Density Estimation** | Gaussian KDE on 200x200 grid via CuPy matrix ops | Yes |
| **H3 Hexagonal Aggregation** | Uber H3 resolution-8 hexagonal spatial binning | — |
| **DBSCAN Spatial Clustering** | Density-based clustering with log-weighted trip counts | — |
| **OD Flow Analysis** | Top 200 origin-destination route pairs | — |
| **Pairwise Distance Matrix** | 263x263 Haversine-approximation via CuPy | Yes |
| **Temporal Patterns** | 168 hour-of-week combinations | — |

### Web Visualization (ArcGIS JS SDK 4.30)

Nine interactive map layers with dark-themed UI:

- **Pickup Hotspots** — Choropleth of trip density by taxi zone
- **Dropoff Density** — Dropoff volume revealing asymmetric trip patterns
- **KDE Heatmap** — GPU-computed kernel density surface showing continuous hotspots
- **H3 Hexagons** — Uniform hexagonal grid aggregation (Uber H3)
- **DBSCAN Clusters** — 53 spatial clusters identified by density-based algorithm
- **OD Flows** — Top 200 busiest routes with thickness-encoded trip volume
- **Temporal Patterns** — Interactive day-hour heatmap of trip volume
- **Revenue by Zone** — Total fare revenue choropleth
- **Tip Analysis** — Average tip patterns across NYC

## Key Findings

- **Midtown Center** is the busiest zone with 139,757 pickups
- **$76.7M** total revenue across all trips in January 2024
- **53 spatial clusters** identified by DBSCAN, concentrated in Manhattan
- **Average trip**: 3.23 miles, 15.7 minutes, $18.16 fare, $3.37 tip
- **Rush hour peaks** clearly visible at 8-9 AM and 5-7 PM on weekdays
- **Weekend nightlife** surge between 11 PM - 2 AM (Friday/Saturday)

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| NYC Yellow Taxi Trips (Jan 2024) | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | ~2.96M rows |
| NYC Taxi Zone Boundaries | [NYC TLC](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip) | 263 zones |

## Project Structure

```
gpu-spatial-analytics/
├── scripts/
│   ├── download_data.py      # Downloads taxi trip data + zone boundaries
│   └── run_analytics.py      # GPU-accelerated analytics pipeline
├── webapp/
│   ├── index.html            # ArcGIS JS SDK interactive map application
│   └── data/                 # Pre-computed GeoJSON analytics output
│       ├── zone_pickups.geojson
│       ├── zone_dropoffs.geojson
│       ├── kde_heatmap.geojson
│       ├── h3_hexagons.geojson
│       ├── dbscan_clusters.geojson
│       ├── od_flows.geojson
│       ├── temporal_patterns.json
│       └── summary.json
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x (for GPU acceleration)
- ~50 MB disk space for data download

### Run the Analytics Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Download NYC Taxi data (~48 MB)
python scripts/download_data.py

# Run GPU-accelerated analytics
python scripts/run_analytics.py
```

### View the Web App

Open `webapp/index.html` in a browser, or serve it locally:

```bash
cd webapp
python -m http.server 8000
# Open http://localhost:8000
```

## Technology Stack

- **NVIDIA CuPy** — GPU-accelerated NumPy-compatible array operations
- **GeoPandas** — Spatial data processing and GeoJSON export
- **Uber H3** — Hexagonal hierarchical spatial indexing
- **scikit-learn** — DBSCAN density-based spatial clustering
- **SciPy** — Statistical analysis and KDE fallback
- **ArcGIS JavaScript SDK 4.30** — Interactive web mapping and visualization
- **Esri Dark Gray Vector Basemap** — Dark-themed cartographic base layer

## GPU Acceleration Details

Two computationally intensive operations leverage CuPy for GPU acceleration on the NVIDIA RTX A4000:

1. **Kernel Density Estimation**: Computing a Gaussian kernel over a 200x200 grid with bandwidth ~800m, evaluating distance to thousands of weighted sample points. The GPU parallelizes the pairwise distance computation across 40,000 grid cells.

2. **Pairwise Distance Matrix**: Computing the 263x263 inter-zone distance matrix using Haversine-approximated Euclidean distance on the GPU, enabling rapid spatial proximity analysis.

When no GPU is available, the pipeline automatically falls back to NumPy/SciPy CPU implementations.

## License

MIT License. Data provided by NYC Taxi & Limousine Commission under [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse).
