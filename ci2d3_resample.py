# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:13:46 2023

@author: adalt043
"""

import pandas as pd

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

# Load CI2D3 data 
ci2d3 = pd.read_csv("D:/Abby/paper_3/CI2D3/ci2d3_1.1b_Abby.csv", index_col=False)

# Convert to datetime
ci2d3["scenedate"] = pd.to_datetime(ci2d3["scenedate"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Define the 2-week intervals
intervals = pd.date_range(start='2008-07-13', end='2013-12-31', freq='2W')

# Add a new column to the DataFrame with the interval labels
ci2d3['interval'] = pd.cut(ci2d3['scenedate'], bins=intervals, labels=range(len(intervals)-1))

# Group by interval and branch then get the first row of each group
ci2d3_unique = ci2d3.groupby(['interval', 'branch']).first().reset_index()

# Drop all rows where there is no unique branch observation within the biweekly interval
ci2d3_unique_branch = ci2d3_unique.dropna(subset=['imgref'])

# Filter DataFrame based on shipping season (July to October)
ci2d3_unique_branch_jaso = ci2d3_unique_branch[(ci2d3_unique_branch['scenedate'].dt.month >= 7) & (ci2d3_unique_branch['scenedate'].dt.month <= 10)]

ci2d3_unique_branch_jaso.to_csv('D:/Abby/paper_3/CI2D3/ci2d3_1.1b_filtered.csv')
