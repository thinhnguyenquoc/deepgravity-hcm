import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import asyncio

# HCMC approximate bounds in WGS84 (EPSG:4326)
HCMC_BOUNDS = {
    'west': 106.6561318,   # min longitude
    'east': 106.66063179999999,   # max longitude  
    'south': 10.7584316,   # min latitude
    'north': 10.7629316    # max latitude
}

def get_hcmc_population(bound, tif_file):
    with rasterio.open(tif_file) as src:
        print(f"File CRS: {src.crs}")
        print(f"File bounds: {src.bounds}")
        
        # Method 1: Using window reading
        window = from_bounds(
            bound['west'], bound['south'],
            bound['east'], bound['north'],
            transform=src.transform
        )
        
        # Read data within HCMC bounds
        hcmc_data = src.read(1, window=window)
        hcmc_transform = src.window_transform(window)
        
        # Handle NoData values (WorldPop typically uses -99999)
        nodata = src.nodata
        print(f"NoData value: {nodata}")
        
        if nodata is not None:
            # Create mask for valid data
            valid_mask = hcmc_data != nodata
            hcmc_population = np.sum(hcmc_data[valid_mask])
            
            print(f"Pixels in HCMC area: {hcmc_data.shape}")
            print(f"Valid pixels: {np.sum(valid_mask)}")
            print(f"Ho Chi Minh City Population (2020): {hcmc_population:,.0f}")
            
            # Additional statistics
            if np.sum(valid_mask) > 0:
                density_data = hcmc_data[valid_mask]
                print(f"Average density: {np.mean(density_data):.2f} people/pixel")
                print(f"Maximum density: {np.max(density_data):.0f} people/pixel")
                print(f"Population range: {np.min(density_data):.0f} - {np.max(density_data):.0f}")
            
            return hcmc_population
        
        else:
            hcmc_population = np.sum(hcmc_data)
            print(f"Ho Chi Minh City Population (2020): {hcmc_population:,.0f}")
            return hcmc_population

# Get HCMC population
# hcmc_data, transform, mask = get_hcmc_population(HCMC_BOUNDS, './data/vnm_ppp_2020.tif')

async def load_grid_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    for district in data:
        cells = district["cells"]
        for cell in cells:
            bound = {}
            bound['west'] = cell["polygon"][0][0]
            bound['east'] = cell["polygon"][1][0]
            bound['south'] = cell["polygon"][0][1]
            bound['north'] = cell["polygon"][2][1]
            print(bound)
            if not hasattr(cell, "population"):
                cell_population = get_hcmc_population(bound, './data/vnm_ppp_2020.tif')
                cell.setdefault("population", str(cell_population))
                print(f"Cell ID: {cell['cell_id']}, Population: {cell['population']}")
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4) # 'indent=4' for pretty-printing
                await asyncio.sleep(2)

# Visualize the results
# plt.figure(figsize=(12, 10))
# if mask is not None:
#     # Show only valid data
#     display_data = np.where(mask, hcmc_data, np.nan)
# else:
#     display_data = hcmc_data

# plt.imshow(display_data, cmap='YlOrRd', 
#            vmax=np.percentile(display_data[~np.isnan(display_data)], 95))
# plt.colorbar(label='Population per pixel')
# plt.title('Ho Chi Minh City Population Distribution (2020)')
# plt.axis('off')
# plt.show()

asyncio.run(load_grid_data('./grid.json'))
