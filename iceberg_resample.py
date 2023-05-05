# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:54:12 2023

@author: adalt043
"""

import pandas as pd

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
iceberg_data = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv", index_col=False)

# Convert to datetime
iceberg_data["datetime_data"] = pd.to_datetime(iceberg_data["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Define the daily intervals
intervals = pd.date_range(start='2012-07-01', end='2019-10-31', freq='D')

# Add a new column to the DataFrame with the interval labels
iceberg_data['interval'] = pd.cut(iceberg_data["datetime_data"], bins=intervals, labels=range(len(intervals)-1))

# Group by interval and branch then get the first row of each group
iceberg_daily = iceberg_data.groupby(['interval', 'beacon_id']).first().reset_index()

# # Drop all rows where there is no unique branch observation within the biweekly interval
# ci2d3_unique_branch = ci2d3_unique.dropna(subset=['imgref'])

# Filter DataFrame based on shipping season (July to October)
iceberg_daily_jaso = iceberg_daily[(iceberg_daily['datetime_data'].dt.month >= 7) & (iceberg_daily['datetime_data'].dt.month <= 10)]

iceberg_daily_jaso.to_csv('D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_04052023_notalbot_daily.csv')
