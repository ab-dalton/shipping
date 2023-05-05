# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:19:19 2023

@author: adalt043
"""

# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------
import geopandas as gpd
import cartopy.feature as cfeature
import pandas as pd
import shapely.wkt
import shapely.geometry
from shapely.geometry import Point
from geopandas import GeoDataFrame
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib import cm

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

path_figures = 'D:/Abby/paper_3/plots/monthly_panels/'

# Load ship data shapefile
ship_data = gpd.read_file("D:/Abby/paper_3/AIS_tracks/SAIS_Tracks_2012to2019_Abby_EasternArctic/SAIS_Tracks_2012to2019_Abby_EasternArctic_nordreg.shp", index_col=False)
ship_data = ship_data.dropna()
ship_data_subset = ship_data.loc[(ship_data['MONTH'] >= 7) & (ship_data['MONTH'] <= 10) & (ship_data['YEAR'] >= 2012) & (ship_data['YEAR'] <= 2019)]

# Load CI2D3 data shapefile
ci2d3 = pd.read_csv("D:/Abby/paper_3/CI2D3/ci2d3_1.1b_filtered.csv", index_col=False)

# Convert to datetime
ci2d3["scenedate"] = pd.to_datetime(ci2d3["scenedate"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Create new column of only identifier
ci2d3['identifier'] = ci2d3.branch.str[-3:]

# -----------------------------------------------------------------------------
# Create grid
# Projection: NAD83 Statistics Canada Lambert (EPSG 3347)
# -----------------------------------------------------------------------------

# Set grid extents - utm mercator values to set corners in m
# Baffin Bay
xmin = 6000000
ymin = 1900000
xmax = 9500000
ymax = 5000000

# # Lancaster Sound
# xmin = 6361000
# ymin = 3891000
# xmax = 6746000
# ymax = 4728000


# Cell size
# Baffin Bay
cell_size = 50000  # cell size in m needs to be divisible by extents above

# Lancaster Sound
# cell_size = 25000

# Create the cells in a loop
grid_cells = []
for x0 in np.arange(xmin, xmax + cell_size, cell_size):
    for y0 in np.arange(ymin, ymax + cell_size, cell_size):
        # Bounds
        x1 = x0 - cell_size
        y1 = y0 + cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
# Set grid projection
grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="epsg:3347")

# Create grid coordinates (cheap centroid)
grid["coords"] = grid["geometry"].apply(lambda x: x.representative_point().coords[:])
grid["coords"] = [coords[0] for coords in grid["coords"]]


# -----------------------------------------------------------------------------
# Plot grid
# -----------------------------------------------------------------------------

# Plot grid
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
grid.plot(
    edgecolor="black",
    color="white",
    ax=ax,
    legend=True,
)

# Observe distortion of grid with WGS84 (EPSG 4326) projection -- 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
grid.to_crs(4326).plot(
    edgecolor="black",
    color="white",
    ax=ax,
    legend=True,
)

# -----------------------------------------------------------------------------
# Create geodataframe and reproject to EPSG 3347
# -----------------------------------------------------------------------------

# Reproject NORDREG zone to match ship tracks
nordreg_poly = gpd.read_file("D:/Abby/paper_3/nordreg/NORDREG_poly.shp")
nordreg_poly = nordreg_poly.to_crs(epsg=3347)

# Create GeoDataFrame from CI2D3 data
geometry = [Point(xy) for xy in zip(ci2d3.lon, ci2d3.lat)]
ci2d3_gdf = GeoDataFrame(ci2d3, crs="epsg:4326", geometry=geometry)

# Reproject CI2D3 data to EPSG 3347
ci2d3_gdf_3347 = ci2d3_gdf.to_crs(epsg=3347)

# Clip CI2D3 database to NORDREG zone to match ship tracks
ci2d3_gdf_clip = gpd.clip(ci2d3_gdf_3347, nordreg_poly)
ci2d3_gdf_clip.plot()

# Reproject ship data to EPSG 3347
ship_gdf = GeoDataFrame(ship_data_subset, crs="epsg:3995")
ship_gdf = ship_gdf.to_crs(epsg=3347)
ship_gdf.plot()


# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

# Define a list of vessel types
vessel_types = ['TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER']

# Create empty dataframes to store the results
mmsi_early = pd.DataFrame()
mmsi_late = pd.DataFrame()

# Loop through each vessel type
for vessel_type in vessel_types:
    
    # Filter ship tracks by vessel type 
    ship_gdf_type = ship_gdf.loc[ship_gdf['NTYPE'] == vessel_type]

    # Merge ship and iceberg geodataframes together
    merged = pd.merge(ship_gdf_type, ci2d3_gdf_clip, how="outer", on='geometry')

    # Spatial join grid with ship tracks
    joined = gpd.sjoin(merged, grid, how="inner", predicate="intersects") #how=inner

    # Filter ship tracks by month
    joined_early = joined.loc[(joined['MONTH'] == 7) | (joined['MONTH'] == 8)]
    joined_late = joined.loc[(joined['MONTH'] == 9) | (joined['MONTH'] == 10)]
    
    # # Filter ship tracks by year
    # joined_early = joined.loc[(joined['YEAR'] >= 2012) & (joined['YEAR'] <= 2015)]
    # joined_late = joined.loc[(joined['YEAR'] >= 2016) & (joined['YEAR'] <= 2019)]

    # Find unique number of ship MMSI per grid cell
    mmsi_early_type = joined_early.groupby(['index_right'])['mmsi'].nunique() 
    mmsi_late_type = joined_late.groupby(['index_right'])['mmsi'].nunique()  

    # Add a new column to store the results for this vessel type
    mmsi_early[vessel_type] = mmsi_early_type
    mmsi_late[vessel_type] = mmsi_late_type
    
# Calculate the total number of unique MMSI for all vessel types combined
mmsi_early['ALL_TYPES'] = mmsi_early.sum(axis=1)
mmsi_late['ALL_TYPES'] = mmsi_late.sum(axis=1)

mmsi_early['PLEASURE VESSELS'].nunique()
mmsi_late['PLEASURE VESSELS'].nunique()

# Merge dataframes to add statistics to the polygon layer
merged_mmsi_early = pd.merge(grid, mmsi_early, left_index=True, right_index=True, how="outer")
merged_mmsi_late = pd.merge(grid, mmsi_late, left_index=True, right_index=True, how="outer")

# CI2D3
# Filter ci2d3 observations by month
joined_early_ci2d3 = joined.loc[(joined['scenedate'].dt.month == 7) | (joined['scenedate'].dt.month == 8)]
joined_late_ci2d3 = joined.loc[(joined['scenedate'].dt.month == 9) | (joined['scenedate'].dt.month == 10)]

joined['branch'].nunique()

# Find number of branch observations per grid cell
ci2d3_early = joined_early_ci2d3.groupby(['index_right'])['branch'].count() 
ci2d3_late = joined_late_ci2d3.groupby(['index_right'])['branch'].count() 
ci2d3_all = joined.groupby(['index_right'])['branch'].count() 

# Merge ci2d3 dataframe to grid - add statistics to the polygon layer
merged_ci2d3_early = pd.merge(grid, ci2d3_early, left_index=True, right_index=True, how="outer")
merged_ci2d3_late = pd.merge(grid, ci2d3_late, left_index=True, right_index=True, how="outer")

# Merge ci2d3 dataframe to grid for entire period - add statistics to the polygon layer
merged_ci2d3 = pd.merge(grid, ci2d3_all, left_index=True, right_index=True, how="outer")


# ----------------------------------------------------------------------------
# Calculate iceberg risk index
# ----------------------------------------------------------------------------

# Merge dataframes with unique number of mmsi and beacon id per grid cell
risk_index_early = pd.merge(merged_mmsi_early, merged_ci2d3_early,  how="outer", on='geometry')
risk_index_late = pd.merge(merged_mmsi_late, merged_ci2d3_late, how="outer", on='geometry')

# Loop through each vessel type column and calculate the risk index
for vessel_type in vessel_types:
    risk_index_early[vessel_type+'_risk'] = risk_index_early[vessel_type] * risk_index_early['beacon_id']
    risk_index_late[vessel_type+'_risk'] = risk_index_late[vessel_type] * risk_index_late['beacon_id']
    
# Sum the risk for all vessel types combined into a column
risk_index_early['total_risk'] = risk_index_early[[vessel_type+'_risk' for vessel_type in vessel_types]].sum(axis=1)
risk_index_late['total_risk'] = risk_index_late[[vessel_type+'_risk' for vessel_type in vessel_types]].sum(axis=1)


risk_index_early['total_risk'] = risk_index_early['total_risk'].replace(0, np.nan)
risk_index_late['total_risk'] = risk_index_late['total_risk'].replace(0, np.nan)

# ----------------------------------------------------------------------------
# Plot grid cells
# ----------------------------------------------------------------------------

# Zoom to Baffin Bay
extents = [-83, -66, 60, 83]
# extents = [-100, -55, 70, 83]

# Set figure DPI
dpi = 300

# Set map projection
proj = ccrs.epsg(3347)

# Set coast
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="darkgrey", facecolor="lightgray", lw=0.75
)

# Set colourbar params
norm = mpl.colors.Normalize(vmin=0, vmax=30) #50

cmap = cm.get_cmap("plasma_r", 30)

##['TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER']
 
fig, axs = plt.subplots(
    1, 2, figsize=(12, 12), constrained_layout=True, subplot_kw={"projection": proj},
)
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)

# Early
axs[0].add_feature(coast)
axs[0].set_extent(extents)
axs[0].set(box_aspect=1)
axs[0].annotate('A', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0].set_facecolor('#D6EAF8')
p1 = merged_ci2d3_early.plot(
    column="branch",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0]
)
for k, spine in axs[0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
gl_2.top_labels = False
gl_2.right_labels = False
gl_2.rotate_labels = False
# rgi.plot(color='white',
#           ax=axs[0,0],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)


# Late
axs[1].add_feature(coast)
axs[1].set_extent(extents) #82.5 
axs[1].set(box_aspect=1)
axs[1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1].set_facecolor('#D6EAF8')
p2 = merged_ci2d3_late.plot(
    column="branch",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1]
)

for k, spine in axs[1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
gl_2.top_labels = False
gl_2.right_labels = False
gl_2.rotate_labels = False
# rgi.plot(color='white',
#           ax=axs[0,1],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)


cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  ax=axs,
                  shrink=0.5,
                  orientation='horizontal') 
cb.ax.tick_params(labelsize=12)
cb.set_label('CI2D3 Observation Count', fontsize=14)


# Save figure
fig.savefig(
    path_figures + "early_late_ci2d3_count.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)




























