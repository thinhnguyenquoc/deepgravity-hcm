import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds


import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Read the WorldPop TIFF file
with rasterio.open('data/vnm_ppp_2020.tif') as src:
    population_data = src.read(1)  # Read first band
    profile = src.profile
    bounds = src.bounds
    crs = src.crs
    
    print(f"Grid shape: {population_data.shape}")
    print(f"CRS: {crs}")
    print(f"Bounds: {bounds}")
    print(f"NoData value: {src.nodata}")

def analyze_worldpop_5km(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        
        # Calculate actual cell size in degrees (approximate for 5km)
        cell_width_deg = transform[0]
        cell_height_deg = abs(transform[4])
        
        print(f"Original transform: {transform}")
        print(f"Cell size (approx): {cell_width_deg:.6f}° x {cell_height_deg:.6f}°")
        
        # Mask no data values
        masked_data = np.ma.masked_where(data == src.nodata, data)
        
        return masked_data, transform, crs

# Usage
population_data, transform, crs = analyze_worldpop_5km('data/vnm_ppp_2020.tif')

def plot_worldpop_data(data, title="WorldPop 5km Grid"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot population data
    im = ax.imshow(data, cmap='YlOrRd', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Population Count')
    
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    plt.show()

plot_worldpop_data(population_data)