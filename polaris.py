# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:15:02 2023

@author: adalt043
"""

import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely
from shapely.geometry import Point
import rasterio
from rasterstats import zonal_stats
import rasterstats as rs


path = "D:/Abby/paper_3/polaris/drive-download-20230412T201907Z-001/POLARIS_RIO_EA_2012_TIF_v04/"

polaris = rxr.open_rasterio("D:/Abby/paper_3/polaris/drive-download-20230412T201907Z-001/POLARIS_RIO_EA_2012_TIF_v04/ea_20120702_polaris_IA_RIO.tif")
polaris_3347 = polaris.rio.reproject("EPSG:3347")
polaris_3347.rio.to_raster("D:/Abby/paper_3/polaris/drive-download-20230412T201907Z-001/POLARIS_RIO_EA_2012_TIF_v04/ea_20120702_polaris_IA_RIO_3347.tif")

polaris2 = rxr.open_rasterio("D:/Abby/paper_3/polaris/drive-download-20230412T201907Z-001/POLARIS_RIO_EA_2012_TIF_v04/ea_20120702_polaris_IA_RIO_3347.tif")

polaris2.plot()

df = dataarray[0].to_pandas()


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
# Calculate zonal stats of raster using grid
# -----------------------------------------------------------------------------

stats = rs.zonal_stats(grid, polaris2, stats=['mean', 'max', 'min', 'median'])





