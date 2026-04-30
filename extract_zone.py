import pandas as pd
import geopandas as gpd

# 1. Load the shapefile directly from the unzipped folder
# Make sure the path matches your actual folder structure
gdf = gpd.read_file("data/taxi_zones/taxi_zones.shp")

# 2. Convert to standard GPS coordinates (WGS84 / EPSG:4326)
gdf = gdf.to_crs("EPSG:4326")

# 3. Calculate the center point (centroid) of each zone
gdf["lon"] = gdf.geometry.centroid.x
gdf["lat"] = gdf.geometry.centroid.y

# 4. Save this as a simple CSV or keep it in memory
zone_coords = gdf[["LocationID", "lat", "lon"]].copy()
zone_coords.to_csv("data/zone_coords.csv", index=False)
print("Saved zone coordinates!")