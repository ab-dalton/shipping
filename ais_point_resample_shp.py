# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:55:59 2023

@author: adalt043
"""

import pandas as pd
import geopandas as gpd

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

# Load AIS data 
ais = gpd.read_file("D:/Abby/paper_3/AIS_tracks/SAIS_eE_2019_06_ManuvPts_Canada/2019_06_pts_nordreg.shp", index_col=False)

# Convert to datetime
ais["YMD_HMS"] = pd.to_datetime(ais["YMD_HMS"].astype(str), format="%Y/%m/%d")

# Define the daily intervals
intervals = pd.date_range(start='2012-07-01', end='2019-12-31', freq='D')

# Add a new column to the DataFrame with the interval labels
ais['interval'] = pd.cut(ais['YMD_HMS'], bins=intervals, include_lowest=True).cat.codes.astype(int)

# Group by interval and mmsi then get the first row of each group
ais_daily_mmsi = ais.groupby(['interval', 'mmsi']).first().reset_index()

# Drop all rows where there is no unique branch observation within the daily interval
ais_daily_mmsi = ais_daily_mmsi.dropna(subset=['YMD_HMS'])

# ais_daily_mmsi['YMD_HMS'] = ais_daily_mmsi['YMD_HMS'].dt.strftime('%Y-%m-%d %H:%M:%S')
# ais_daily_mmsi.to_file('D:/Abby/paper_3/AIS_tracks/2019_06_daily_pts.shp')




# Adding in vessel type to point data if it's missing 

# Load tracks into a DataFrame
df = pd.read_csv("D:/Abby/paper_3/AIS_tracks/SAIS_Tracks_2012to2019_Abby_EasternArctic/SAIS_Tracks_2012to2019_Abby_EasternArctic_nordreg.csv", encoding='latin-1', index_col=False)

# Create a new DataFrame containing only the MMSI and vessel type columns
df_unique = df[['mmsi', 'NTYPE']]

# Drop duplicates
df_unique = df_unique.drop_duplicates()

# Convert the DataFrame to a dictionary
unique_dict = df_unique.set_index('mmsi')['NTYPE'].to_dict()

ais_daily_mmsi['NTYPE'] = ais_daily_mmsi['mmsi'].map(unique_dict)


