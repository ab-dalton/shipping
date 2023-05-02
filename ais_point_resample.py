# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:58:58 2023

@author: adalt043
"""

import pandas as pd

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

# Load CI2D3 data 
ais = pd.read_csv("D:/Abby/paper_3/AIS_tracks/SAIS_eE_2019_06_ManuvPts_Canada/2019_06_pts_nordreg.csv", index_col=False)

# Convert to datetime
ais["YMD_HMS"] = pd.to_datetime(ais["YMD_HMS"].astype(str), format="%d/%m/%Y")

# Define the daily intervals
intervals = pd.date_range(start='2012-07-01', end='2019-12-31', freq='D')

# Add a new column to the DataFrame with the interval labels
ais['interval'] = pd.cut(ais['YMD_HMS'], bins=intervals, labels=range(len(intervals)-1))

# Group by interval and mmsi then get the first row of each group
ais_daily_mmsi = ais.groupby(['interval', 'mmsi']).first().reset_index()

# Drop all rows where there is no unique branch observation within the daily interval
ais_daily_mmsi = ais_daily_mmsi.dropna(subset=['YMD_HMS'])

ais_daily_mmsi.to_csv('D:/Abby/paper_3/AIS_tracks/2019_06_daily_pts.csv')


