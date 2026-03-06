"""Download NYC Taxi trip data and taxi zone boundaries."""
import os
import requests
import zipfile
import io

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Download NYC Taxi Zone shapefile (boundaries for spatial joins)
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
zones_path = os.path.join(DATA_DIR, "taxi_zones")

if not os.path.exists(zones_path):
    print("Downloading NYC Taxi Zones shapefile...")
    r = requests.get(ZONES_URL)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(zones_path)
    print(f"  Extracted to {zones_path}")
else:
    print("Taxi zones already downloaded.")

# 2. Download NYC Yellow Taxi Trip Data (Jan 2024 - Parquet, ~45MB)
TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
trips_path = os.path.join(DATA_DIR, "yellow_tripdata_2024-01.parquet")

if not os.path.exists(trips_path):
    print("Downloading NYC Yellow Taxi trips (Jan 2024)...")
    r = requests.get(TRIPS_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(trips_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB", end="\r")
    print(f"\n  Saved to {trips_path}")
else:
    print("Trip data already downloaded.")

# 3. Download NYC Borough boundaries (for context in the map)
BOROUGHS_URL = "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&type=GeoJSON"
boroughs_path = os.path.join(DATA_DIR, "nyc_boroughs.geojson")

if not os.path.exists(boroughs_path):
    print("Downloading NYC Borough boundaries...")
    r = requests.get(BOROUGHS_URL)
    r.raise_for_status()
    with open(boroughs_path, "w") as f:
        f.write(r.text)
    print(f"  Saved to {boroughs_path}")
else:
    print("Borough boundaries already downloaded.")

print("\nAll data downloaded successfully!")
