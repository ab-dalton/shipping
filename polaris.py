# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:15:02 2023

@author: adalt043
"""

from geopandas import GeoDataFrame
import geopandas as gpd
import os
import pandas as pd


# -----------------------------------------------------------------------------
# Load all shapefile data and concat
# -----------------------------------------------------------------------------

# Set the directory path where the shapefiles are located
folder_path = "D:/Abby/paper_3/polaris/50km_points/"

# Create an empty list to store the GeoDataFrames
gdfs = []

# Loop through each shapefile in the directory and read it into a GeoDataFrame
for file in os.listdir(folder_path):
    if file.endswith(".shp"):
        filepath = os.path.join(folder_path, file)
        gdf = gpd.read_file(filepath)
        # Add a new column to the GeoDataFrame with the filename as the value
        gdf['filename'] = file[:-4]
        # Append the GeoDataFrame to the list
        gdfs.append(gdf)

# Concatenate all the GeoDataFrames into a single GeoDataFrame
merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdf.crs)

gdf = GeoDataFrame(merged_gdf, crs="epsg:3995")
gdf_3347 = merged_gdf.to_crs(epsg=3347)

# Clip sea ice database to NCAA zone 
ncaa_poly = gpd.read_file("D:/Abby/paper_3/polaris/ncaa_poly.shp")
ncaa_poly = ncaa_poly.to_crs(epsg=3347)
merged_clip = gpd.clip(gdf_3347, ncaa_poly)
merged_clip.plot()

# Split filename into columns for date and ice class
merged_clip['date'] = merged_clip['filename'].str.split('_').str[1]
merged_clip['ice_class'] = merged_clip['filename'].str.split('_').str[3] + merged_clip['filename'].str.split('_').str[4]
merged_clip['ice_class'] = merged_clip['ice_class'].str.replace('RIO', '')

# Convert date column to datetime

merged_clip['date'] = pd.to_datetime(
    merged_clip['date'].astype(str), format="%Y%m%d"
)


merged_clip.to_csv("D:/Abby/paper_3/polaris/polaris_clipped.csv")


