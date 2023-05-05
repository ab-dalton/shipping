# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:55:59 2023

@author: adalt043
"""

import pandas as pd
import geopandas as gpd
import os


# -----------------------------------------------------------------------------
# Resample multiple files in a folder
# -----------------------------------------------------------------------------

# Define the daily intervals
intervals = pd.date_range(start='2012-07-01', end='2019-10-31', freq='D')

# Create an empty list to store the geodataframes
gdfs = []

# Loop through the folder containing the shapefiles
folder_path = 'D:/Abby/paper_3/AIS_tracks/ais_points_nordreg/'
for shp_file in os.listdir(folder_path):
    if shp_file.endswith('.shp'):
        # Load the shapefile and convert the datetime column
        ais = gpd.read_file(os.path.join(folder_path, shp_file), index_col=False)
        ais["YMD_HMS"] = pd.to_datetime(ais["YMD_HMS"].astype(str), format="%Y/%m/%d")
        
        # Add a new column to the DataFrame with the interval labels
        ais['interval'] = pd.cut(ais['YMD_HMS'], bins=intervals, include_lowest=True).cat.codes.astype(int)
        
        # Group by interval and mmsi then get the first row of each group
        ais_daily_mmsi = ais.groupby(['interval', 'mmsi']).first().reset_index()
        
        # Append the GeoDataFrame to the list
        gdfs.append(ais_daily_mmsi)
        
# Concatenate all the GeoDataFrames into a single GeoDataFrame
merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ais_daily_mmsi.crs)

merged_filtered = merged_gdf.loc[(merged_gdf['mmsi'] >= 201000000) & (merged_gdf['mmsi'] <= 775999999)]

merged_filtered.to_csv("D:/Abby/paper_3/AIS_tracks/ais_points_nordreg/ais_points_nordreg_merged_daily.csv")








# # -----------------------------------------------------------------------------
# # Resample single file
# # -----------------------------------------------------------------------------

# # Load AIS data 
# ais = gpd.read_file("D:/Abby/paper_3/AIS_tracks/SAIS_eE_2019_06_ManuvPts_Canada/2019_06_pts_nordreg.shp", index_col=False)

# # Convert to datetime
# ais["YMD_HMS"] = pd.to_datetime(ais["YMD_HMS"].astype(str), format="%Y/%m/%d")

# # Define the daily intervals
# intervals = pd.date_range(start='2012-07-01', end='2019-12-31', freq='D')

# # Add a new column to the DataFrame with the interval labels
# ais['interval'] = pd.cut(ais['YMD_HMS'], bins=intervals, include_lowest=True).cat.codes.astype(int)

# # Group by interval and mmsi then get the first row of each group
# ais_daily_mmsi = ais.groupby(['interval', 'mmsi']).first().reset_index()

# # Drop all rows where there is no unique branch observation within the daily interval
# ais_daily_mmsi = ais_daily_mmsi.dropna(subset=['YMD_HMS'])


# ais_daily_mmsi['YMD_HMS'] = ais_daily_mmsi['YMD_HMS'].dt.strftime('%Y-%m-%d %H:%M:%S')
# ais_daily_mmsi.to_file('D:/Abby/paper_3/AIS_tracks/2019_06_daily_pts.shp')












