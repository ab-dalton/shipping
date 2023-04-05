# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:05:55 2023

@author: adalt043
"""

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
ship_data_subset = ship_data.loc[(ship_data['MONTH'] >= 7) & (ship_data['MONTH'] <= 10) & (ship_data['YEAR'] >= 2012) & (ship_data['YEAR'] <= 2019)]

# Load most recent Iceberg Beacon Database output file
iceberg_data = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv", index_col=False)

# Convert to datetime
iceberg_data["datetime_data"] = pd.to_datetime(iceberg_data["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Filter iceberg database to shipping season (July-October)
iceberg_data_subset = iceberg_data[(iceberg_data['datetime_data'].dt.month >= 7) & (iceberg_data['datetime_data'].dt.month <= 10) & (iceberg_data['datetime_data'].dt.year >= 2012) & (iceberg_data['datetime_data'].dt.year <= 2019)]

# # Load CI2D3 data shapefile
# ci2d3 = pd.read_csv("D:/Abby/paper_3/CI2D3/CCIN12678_20181113_CI2D3_01.csv", index_col=False)
# ci2d3 = ci2d3.dropna()

# # Convert to datetime
# ci2d3["scenedate"] = pd.to_datetime(ci2d3["scenedate"].astype(str), format="%Y-%m-%d %H:%M:%S")

# # Filter CI2D3 database to shipping season (July-October)
# ci2d3_subset = ci2d3[(ci2d3['scenedate'].dt.month >= 7) & (ci2d3['scenedate'].dt.month <= 10)]

# # Create new column of only identifier
# ci2d3_subset['identifier'] = ci2d3_subset.lineage.str[-3:]

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

## Iceberg Database
# Create GeoDataFrame from iceberg data
geometry = [Point(xy) for xy in zip(iceberg_data_subset.longitude, iceberg_data_subset.latitude)]
gdf = GeoDataFrame(iceberg_data_subset, crs="epsg:4326", geometry=geometry)

# Reproject iceberg data to EPSG 3347
iceberg_gdf = gdf.to_crs(epsg=3347)

# # Create GeoDataFrame from CI2D3 data
# geometry = [Point(xy) for xy in zip(ci2d3_subset.lon, ci2d3_subset.lat)]
# gdf = GeoDataFrame(ci2d3_subset, crs="epsg:4326", geometry=geometry)

# # Reproject CI2D3 data to EPSG 3347
# ci2d3_gdf = gdf.to_crs(epsg=3347)

# Reproject ship data to EPSG 3347
ship_gdf = GeoDataFrame(ship_data_subset, crs="epsg:3995")
ship_gdf = ship_gdf.to_crs(epsg=3347)

# Clip iceberg database to NORDREG zone to match ship tracks
nordreg_poly = gpd.read_file("D:/Abby/paper_3/nordreg/NORDREG_poly.shp")
nordreg_poly = nordreg_poly.to_crs(epsg=3347)
iceberg_gdf_clip = gpd.clip(iceberg_gdf, nordreg_poly)
iceberg_gdf_clip.plot()

# # Clip CI2D3 database to NORDREG zone to match ship tracks
# ci2d3_gdf_clip = gpd.clip(ci2d3_gdf, nordreg_poly)
# ci2d3_gdf_clip.plot()

# Merge ship and iceberg geodataframes together
merged = pd.merge(ship_gdf, iceberg_gdf_clip, how="outer", on='geometry')

# Merge ship, iceberg, and CI2D3 geodataframes together
# merged_ci2d3 = pd.merge(merged, ci2d3_gdf_clip, how="outer", on='geometry')

# Spatial join grid with ship tracks
spatial_joined = gpd.sjoin(merged, grid, how="inner", op="intersects")


# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

# Filter ship tracks by vessel type 
# ('TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER')
spatial_joined = spatial_joined.loc[spatial_joined['NTYPE'] == "FISHING"]


# Filter ship tracks by month
mmsi_joined_july = spatial_joined.loc[spatial_joined['MONTH'] == 7]
mmsi_joined_aug = spatial_joined.loc[spatial_joined['MONTH'] == 8]
mmsi_joined_sept = spatial_joined.loc[spatial_joined['MONTH'] == 9]
mmsi_joined_oct = spatial_joined.loc[spatial_joined['MONTH'] == 10]

# Find unique number of ship MMSI per grid cell
mmsi_july = mmsi_joined_july.groupby(['index_right'])['mmsi'].nunique() 
mmsi_aug = mmsi_joined_aug.groupby(['index_right'])['mmsi'].nunique()  
mmsi_sept = mmsi_joined_sept.groupby(['index_right'])['mmsi'].nunique() 
mmsi_oct = mmsi_joined_oct.groupby(['index_right'])['mmsi'].nunique() 

# Merge dataframes to add statistics to the polygon layer
merged_mmsi_july = pd.merge(grid, mmsi_july, left_index=True, right_index=True, how="outer")
merged_mmsi_aug = pd.merge(grid, mmsi_aug, left_index=True, right_index=True, how="outer")
merged_mmsi_sept = pd.merge(grid, mmsi_sept, left_index=True, right_index=True, how="outer")
merged_mmsi_oct = pd.merge(grid, mmsi_oct, left_index=True, right_index=True, how="outer")

## Icebergs
# Filter iceberg tracks by month
beacon_joined_july = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 7]
beacon_joined_aug = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 8]
beacon_joined_sept = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 9]
beacon_joined_oct = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 10]

# Find unique number of iceberg beacon IDs per grid cell
beaconid_july = beacon_joined_july.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_aug = beacon_joined_aug.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_sept = beacon_joined_sept.groupby(['index_right'])['beacon_id'].nunique() 
beaconid_oct = beacon_joined_oct.groupby(['index_right'])['beacon_id'].nunique() 

# Merge dataframes to add statistics to the polygon layer
merged_beaconid_july = pd.merge(grid, beaconid_july, left_index=True, right_index=True, how="outer")
merged_beaconid_aug = pd.merge(grid, beaconid_aug, left_index=True, right_index=True, how="outer")
merged_beaconid_sept = pd.merge(grid, beaconid_sept, left_index=True, right_index=True, how="outer")
merged_beaconid_oct = pd.merge(grid, beaconid_oct, left_index=True, right_index=True, how="outer")


# Define the months to iterate over
months = [7, 8, 9, 10]

# Initialize empty dictionaries to store the dataframes
mmsi_dict = {}
beaconid_dict = {}
# ci2d3_dict = {}

# Loop over the months and filter dataframes
for month in months:
    mmsi_joined = spatial_joined.loc[spatial_joined['MONTH'] == month]
    beacon_joined = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == month]
    # ci2d3_joined = spatial_joined.loc[spatial_joined['scenedate'].dt.month == month]

    # Find unique number of ship MMSI, iceberg beacon IDs, and CI2D3 lineage per grid cell
    mmsi = mmsi_joined.groupby(['index_right'])['mmsi'].nunique() 
    beaconid = beacon_joined.groupby(['index_right'])['beacon_id'].nunique() 
    # ci2d3 = ci2d3_joined.groupby(['index_right'])['identifier'].nunique() 

    # Merge dataframes to add statistics to the polygon layer
    merged_mmsi = pd.merge(grid, mmsi, left_index=True, right_index=True, how="outer")
    merged_beaconid = pd.merge(grid, beaconid, left_index=True, right_index=True, how="outer")
    # merged_ci2d3 = pd.merge(grid, ci2d3, left_index=True, right_index=True, how="outer")

    # Add the merged dataframes to the dictionaries
    mmsi_dict[month] = merged_mmsi
    beaconid_dict[month] = merged_beaconid
    # ci2d3_dict[month] = merged_ci2d3

merged_mmsi_july = mmsi_dict[7]
merged_mmsi_aug = mmsi_dict[8]
merged_mmsi_sept = mmsi_dict[9]
merged_mmsi_oct = mmsi_dict[10]

merged_beaconid_july = beaconid_dict[7]
merged_beaconid_aug = beaconid_dict[8]
merged_beaconid_sept = beaconid_dict[9]
merged_beaconid_oct = beaconid_dict[10]

# merged_ci2d3_july = ci2d3_dict[7]
# merged_ci2d3_aug = ci2d3_dict[8]
# merged_ci2d3_sept = ci2d3_dict[9]
# merged_ci2d3_oct = ci2d3_dict[10]



# ----------------------------------------------------------------------------
# Calculate iceberg risk index
# ----------------------------------------------------------------------------

# Merge dataframes with unique number of mmsi and beacon id per grid cell
risk_index_july = pd.merge(merged_mmsi_july, merged_beaconid_july,  how="outer", on='geometry')
risk_index_aug = pd.merge(merged_mmsi_aug, merged_beaconid_aug, how="outer", on='geometry')
risk_index_sept = pd.merge(merged_mmsi_sept, merged_beaconid_sept, how="outer", on='geometry')
risk_index_oct = pd.merge(merged_mmsi_oct, merged_beaconid_oct, how="outer", on='geometry')

# # Merge ships+icebergs above with unique number of ci2d3 lineage per grid cell
# risk_index_july = pd.merge(risk_july, merged_ci2d3_july,  how="outer", on='geometry')
# risk_index_aug = pd.merge(risk_aug, merged_ci2d3_aug, how="outer", on='geometry')
# risk_index_sept = pd.merge(risk_sept, merged_ci2d3_sept, how="outer", on='geometry')
# risk_index_oct = pd.merge(risk_oct, merged_ci2d3_oct, how="outer", on='geometry')

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

# Lancaster Sound
# extents = [-90, -70, 70, 76]

# Set figure DPI
dpi = 300

# Set map projection
proj = ccrs.epsg(3347)

# Set coast
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="darkgrey", facecolor="lightgray", lw=0.75
)

# Set colourbar params
norm = mpl.colors.Normalize(vmin=0, vmax=200) #50
cmap = cm.get_cmap("plasma_r", 10)


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
axs[0, 0].set_extent(extents)
axs[0, 0].set(box_aspect=1)
axs[0, 0].annotate('A', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,0].set_facecolor('#D6EAF8')
p1 = risk_index_july.plot(
    column="risk_index",
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
axs[0, 1].set_extent(extents)
axs[0, 1].set(box_aspect=1)
axs[0, 1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,1].set_facecolor('#D6EAF8')
p2 = risk_index_aug.plot(
    column="risk_index",
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
axs[1, 0].set_extent(extents)
axs[1, 0].set(box_aspect=1)
axs[1, 0].annotate('C', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1,0].set_facecolor('#D6EAF8')
p3 = risk_index_sept.plot(
    column="risk_index",
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
axs[1, 1].set_extent(extents)
axs[1, 1].set(box_aspect=1)
axs[1, 1].annotate('D', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1,1].set_facecolor('#D6EAF8')
p4 = risk_index_oct.plot(
    column="risk_index",
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
cb.set_label('Iceberg-Ship Risk Index: 2012-2019', fontsize=14)


# Save figure
fig.savefig(
    path_figures + "jaso_2012_2019_risk.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)





























# This code finds the iceberg and ship stats NOT in a loop

# ## Ships

# # Filter ship tracks by vessel type 
# # ('TANKER','FISHING','GOVERNMENT/RESEARCH','CARGO','PLEASURE VESSELS','FERRY/RO-RO/PASSENGER','OTHERS/SPECIAL SHIPS','DRY BULK','TUGS/PORT','CONTAINER')
# # spatial_joined = spatial_joined.loc[spatial_joined['NTYPE'] == "FISHING"]

# # Filter ship tracks by month
# mmsi_joined_july = spatial_joined.loc[spatial_joined['MONTH'] == 7]
# mmsi_joined_aug = spatial_joined.loc[spatial_joined['MONTH'] == 8]
# mmsi_joined_sept = spatial_joined.loc[spatial_joined['MONTH'] == 9]
# mmsi_joined_oct = spatial_joined.loc[spatial_joined['MONTH'] == 10]

# # Find unique number of ship MMSI per grid cell
# mmsi_july = mmsi_joined_july.groupby(['index_right'])['mmsi'].nunique() 
# mmsi_aug = mmsi_joined_aug.groupby(['index_right'])['mmsi'].nunique()  
# mmsi_sept = mmsi_joined_sept.groupby(['index_right'])['mmsi'].nunique() 
# mmsi_oct = mmsi_joined_oct.groupby(['index_right'])['mmsi'].nunique() 

# # Merge dataframes to add statistics to the polygon layer
# merged_mmsi_july = pd.merge(grid, mmsi_july, left_index=True, right_index=True, how="outer")
# merged_mmsi_aug = pd.merge(grid, mmsi_aug, left_index=True, right_index=True, how="outer")
# merged_mmsi_sept = pd.merge(grid, mmsi_sept, left_index=True, right_index=True, how="outer")
# merged_mmsi_oct = pd.merge(grid, mmsi_oct, left_index=True, right_index=True, how="outer")

# ## Icebergs
# # Filter iceberg tracks by month
# beacon_joined_july = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 7]
# beacon_joined_aug = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 8]
# beacon_joined_sept = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 9]
# beacon_joined_oct = spatial_joined.loc[spatial_joined['datetime_data'].dt.month == 10]

# # Find unique number of iceberg beacon IDs per grid cell
# beaconid_july = beacon_joined_july.groupby(['index_right'])['beacon_id'].nunique() 
# beaconid_aug = beacon_joined_aug.groupby(['index_right'])['beacon_id'].nunique() 
# beaconid_sept = beacon_joined_sept.groupby(['index_right'])['beacon_id'].nunique() 
# beaconid_oct = beacon_joined_oct.groupby(['index_right'])['beacon_id'].nunique() 

# # Merge dataframes to add statistics to the polygon layer
# merged_beaconid_july = pd.merge(grid, beaconid_july, left_index=True, right_index=True, how="outer")
# merged_beaconid_aug = pd.merge(grid, beaconid_aug, left_index=True, right_index=True, how="outer")
# merged_beaconid_sept = pd.merge(grid, beaconid_sept, left_index=True, right_index=True, how="outer")
# merged_beaconid_oct = pd.merge(grid, beaconid_oct, left_index=True, right_index=True, how="outer")

# ## CI2D3
# # Filter iceberg tracks by month
# ci2d3_joined_july = spatial_joined.loc[spatial_joined['scenedate'].dt.month == 7]
# ci2d3_joined_aug = spatial_joined.loc[spatial_joined['scenedate'].dt.month == 8]
# ci2d3_joined_sept = spatial_joined.loc[spatial_joined['scenedate'].dt.month == 9]
# ci2d3_joined_oct = spatial_joined.loc[spatial_joined['scenedate'].dt.month == 10]

# # Find unique number of iceberg beacon IDs per grid cell
# ci2d3_july = ci2d3_joined_july.groupby(['index_right'])['lineage'].nunique() 
# ci2d3_aug = ci2d3_joined_aug.groupby(['index_right'])['lineage'].nunique() 
# ci2d3_sept = ci2d3_joined_sept.groupby(['index_right'])['lineage'].nunique() 
# ci2d3_oct = ci2d3_joined_oct.groupby(['index_right'])['lineage'].nunique() 

# # Merge dataframes to add statistics to the polygon layer
# merged_ci2d3_july = pd.merge(grid, ci2d3_july, left_index=True, right_index=True, how="outer")
# merged_ci2d3_aug = pd.merge(grid, ci2d3_aug, left_index=True, right_index=True, how="outer")
# merged_ci2d3_sept = pd.merge(grid, ci2d3_sept, left_index=True, right_index=True, how="outer")
# merged_ci2d3_oct = pd.merge(grid, ci2d3_oct, left_index=True, right_index=True, how="outer")
























