# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:15:46 2023

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
ship_data_subset = ship_data.loc[(ship_data['MONTH'] >= 7) & (ship_data['MONTH'] <= 10) & (ship_data['YEAR'] >= 2016) & (ship_data['YEAR'] <= 2019)]

# Load most recent Iceberg Beacon Database output file
iceberg_data = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv", index_col=False)

# Convert to datetime
iceberg_data["datetime_data"] = pd.to_datetime(iceberg_data["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Filter iceberg database to shipping season (July-October)
iceberg_data_subset = iceberg_data[(iceberg_data['datetime_data'].dt.month >= 7) & (iceberg_data['datetime_data'].dt.month <= 10) & (iceberg_data['datetime_data'].dt.year >= 2016) & (iceberg_data['datetime_data'].dt.year <= 2019)]

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

# Cell size
# Baffin Bay
cell_size = 50000  # cell size in m needs to be divisible by extents above

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

# Create GeoDataFrame from iceberg data
geometry = [Point(xy) for xy in zip(iceberg_data_subset.longitude, iceberg_data_subset.latitude)]
gdf = GeoDataFrame(iceberg_data_subset, crs="epsg:4326", geometry=geometry)

# Reproject iceberg data to EPSG 3347
iceberg_gdf = gdf.to_crs(epsg=3347)

# Clip iceberg database to NORDREG zone to match ship tracks
nordreg_poly = gpd.read_file("D:/Abby/paper_3/nordreg/NORDREG_poly.shp")
nordreg_poly = nordreg_poly.to_crs(epsg=3347)
iceberg_gdf_clip = gpd.clip(iceberg_gdf, nordreg_poly)
iceberg_gdf_clip.plot()

# Reproject ship data to EPSG 3347
ship_gdf = GeoDataFrame(ship_data_subset, crs="epsg:3995")
ship_gdf = ship_gdf.to_crs(epsg=3347)
ship_gdf.plot()

# Filter ship tracks by vessel type 
# vessel_type = ['TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER']
ship_gdf = ship_gdf.loc[ship_gdf['NTYPE'] == 'TUGS/PORT']

# Merge ship and iceberg geodataframes together
merged = pd.merge(ship_gdf, iceberg_gdf_clip, how="outer", on='geometry')

# Spatial join grid with ship tracks
joined = gpd.sjoin(merged, grid, how="inner", predicate="intersects") #how=inner

# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

## Ships

# Filter ship tracks by month
joined_july = joined.loc[joined['MONTH'] == 7]
joined_aug = joined.loc[joined['MONTH'] == 8]
joined_sept = joined.loc[joined['MONTH'] == 9]
joined_oct = joined.loc[joined['MONTH'] == 10]

# Find unique number of ship MMSI per grid cell
mmsi_july = joined_july.groupby(['index_right'])['mmsi'].nunique() 
mmsi_aug = joined_aug.groupby(['index_right'])['mmsi'].nunique()  
mmsi_sept = joined_sept.groupby(['index_right'])['mmsi'].nunique() 
mmsi_oct = joined_oct.groupby(['index_right'])['mmsi'].nunique() 

# Merge dataframes to add statistics to the polygon layer
merged_mmsi_july = pd.merge(grid, mmsi_july, left_index=True, right_index=True, how="outer")
merged_mmsi_aug = pd.merge(grid, mmsi_aug, left_index=True, right_index=True, how="outer")
merged_mmsi_sept = pd.merge(grid, mmsi_sept, left_index=True, right_index=True, how="outer")
merged_mmsi_oct = pd.merge(grid, mmsi_oct, left_index=True, right_index=True, how="outer")


## Icebergs
# Filter iceberg tracks by month
joined_july = joined.loc[joined['datetime_data'].dt.month == 7]
joined_aug = joined.loc[joined['datetime_data'].dt.month == 8]
joined_sept = joined.loc[joined['datetime_data'].dt.month == 9]
joined_oct = joined.loc[joined['datetime_data'].dt.month == 10]

# Find unique number of iceberg beacon IDs per grid cell
beaconid_july = joined_july.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_aug = joined_aug.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_sept = joined_sept.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_oct = joined_oct.groupby(['index_right'])['beacon_id'].nunique() 

# Merge dataframes to add statistics to the polygon layer
merged_beaconid_july = pd.merge(grid, beaconid_july, left_index=True, right_index=True, how="outer")
merged_beaconid_aug = pd.merge(grid, beaconid_aug, left_index=True, right_index=True, how="outer")
merged_beaconid_sept = pd.merge(grid, beaconid_sept, left_index=True, right_index=True, how="outer")
merged_beaconid_oct = pd.merge(grid, beaconid_oct, left_index=True, right_index=True, how="outer")


# ----------------------------------------------------------------------------
# Calculate iceberg risk index
# ----------------------------------------------------------------------------

# Merge dataframes with unique number of mmsi and beacon id per grid cell
risk_index_july = pd.merge(merged_mmsi_july, merged_beaconid_july,  how="outer", on='geometry')
risk_index_aug = pd.merge(merged_mmsi_aug, merged_beaconid_aug, how="outer", on='geometry')
risk_index_sept = pd.merge(merged_mmsi_sept, merged_beaconid_sept, how="outer", on='geometry')
risk_index_oct = pd.merge(merged_mmsi_oct, merged_beaconid_oct, how="outer", on='geometry')

# Calculate risk index
risk_index_july['risk_index'] = risk_index_july['mmsi'] * risk_index_july['beacon_id']
risk_index_aug['risk_index'] = risk_index_aug['mmsi'] * risk_index_aug['beacon_id']
risk_index_sept['risk_index'] = risk_index_sept['mmsi'] * risk_index_sept['beacon_id']
risk_index_oct['risk_index'] = risk_index_oct['mmsi'] * risk_index_oct['beacon_id']

# ----------------------------------------------------------------------------
# Plot grid cells
# ----------------------------------------------------------------------------

# Zoom to Baffin Bay
# extents = [-83, -64, 58, 83]
extents = [-100, -55, 60, 85]

# Set figure DPI
dpi = 300

# Set map projection
proj = ccrs.epsg(3347)

# Set coast
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="darkgrey", facecolor="lightgray", lw=0.75
)

# Set colourbar params
norm = mpl.colors.Normalize(vmin=0, vmax=50) #50

cmap = cm.get_cmap("plasma_r", 20)


fig, axs = plt.subplots(
    2, 2, figsize=(12, 12), constrained_layout=True, subplot_kw={"projection": proj},
)
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)

# July
axs[0, 0].add_feature(coast)
axs[0, 0].set_extent([-100, -55, 55, 87])
axs[0, 0].set(box_aspect=1)
axs[0, 0].annotate('A', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,0].set_facecolor('#D6EAF8')
p1 = merged_mmsi_july.plot(
    column="mmsi",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0, 0]
)
for k, spine in axs[0,0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[0, 0].gridlines(
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


# August
axs[0, 1].add_feature(coast)
axs[0, 1].set_extent([-100, -55, 55, 87]) #82.5 
axs[0, 1].set(box_aspect=1)
axs[0, 1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,1].set_facecolor('#D6EAF8')
p2 = merged_mmsi_aug.plot(
    column="mmsi",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0 ,1]
)

for k, spine in axs[0,1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[0, 1].gridlines(
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


# September
axs[1, 0].add_feature(coast)
axs[1, 0].set_extent([-100, -55, 55, 87])
axs[1, 0].set(box_aspect=1)
axs[1, 0].annotate('C', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1,0].set_facecolor('#D6EAF8')
p3 = merged_mmsi_aug.plot(
    column="mmsi",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1, 0]
)
for k, spine in axs[1,0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[1,0].gridlines(
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
#           ax=axs[1,0],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)


# October
axs[1, 1].add_feature(coast)
axs[1, 1].set_extent([-100, -55, 60, 82.5])
axs[1, 1].set(box_aspect=1)
axs[1, 1].annotate('D', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1,1].set_facecolor('#D6EAF8')
p4 = merged_mmsi_sept.plot(
    column="mmsi",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1 ,1]
)
for k, spine in axs[1,1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[1, 1].gridlines(
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
#           ax=axs[1,1],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)

cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                  ax=axs,
                  shrink=0.5,
                  orientation='horizontal') 
cb.ax.tick_params(labelsize=12)
cb.set_label('Unique # of MMSI: 2016-2019, Tugs/Port', fontsize=14)


# Save figure
fig.savefig(
    path_figures + "jaso_2016_2019_tugsport.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)

























