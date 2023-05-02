# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:34:41 2023

@author: adalt043
"""


# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame


#------------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

path_figures = 'D:/Abby/paper_3/plots/monthly_panels/'

# Load ship data shapefile
ship_data = pd.read_csv("D:/Abby/paper_3/AIS_tracks/SAIS_Tracks_2012to2019_Abby_EasternArctic/SAIS_Tracks_2012to2019_Abby_EasternArctic_nordreg.csv", index_col=False, encoding='latin-1')
ship_data = ship_data.dropna()


# group the data by the column you want to use for the stacked bar plot, and then count the number of unique values in each group
grouped = ship_data.groupby(['YEAR', 'NTYPE'],as_index=False)['mmsi'].nunique()

# pivot the data to create a stacked bar chart
stacked_data = grouped.pivot(index='YEAR', columns='NTYPE', values='mmsi')

vessel_type = ('TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER')
colors = sns.color_palette("colorblind", n_colors=len(vessel_type))
cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

fig, ax = plt.subplots(figsize=(12, 8))
stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap=cmap1)
ax.set(ylabel="Unique # of MMSI", xlabel="Year")
ax.legend(title='', frameon=False)

fig.savefig(
    path_figures + "unique_mmsi_bars",
    dpi=300,
    transparent=False,
    bbox_inches="tight",
)



# create a new dataframe to store the results
ship_counts = pd.DataFrame(columns=['Ship Type', 'Year', 'Count'])

# loop through the columns and index of the pivoted data frame, and add the results to the new dataframe
for ship_type in stacked_data.columns:
    for year in stacked_data.index:
        count = stacked_data.loc[year, ship_type]
        ship_counts = ship_counts.append({'Ship Type': ship_type, 'Year': year, 'Count': count}, ignore_index=True)
        
        
ship_counts = pd.pivot_table(grouped, values='mmsi', index='NTYPE', columns='YEAR', aggfunc='sum')

ship_counts.to_csv('D:/Abby/paper_3/AIS_tracks/ship_counts.csv')
