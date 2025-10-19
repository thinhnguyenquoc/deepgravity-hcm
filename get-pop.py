

# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from rasterio.windows import from_bounds
# import geopandas as gpd
# from shapely.geometry import box

# # HCMC approximate bounds in WGS84 (EPSG:4326)
# HCMC_BOUNDS = {
#     'west': 106.3,   # min longitude
#     'east': 107.0,   # max longitude  
#     'south': 10.3,   # min latitude
#     'north': 11.1    # max latitude
# }

# def get_hcmc_population(tif_file):
#     with rasterio.open(tif_file) as src:
#         print(f"File CRS: {src.crs}")
#         print(f"File bounds: {src.bounds}")
        
#         # Method 1: Using window reading
#         window = from_bounds(
#             HCMC_BOUNDS['west'], HCMC_BOUNDS['south'],
#             HCMC_BOUNDS['east'], HCMC_BOUNDS['north'],
#             transform=src.transform
#         )
        
#         # Read data within HCMC bounds
#         hcmc_data = src.read(1, window=window)
#         hcmc_transform = src.window_transform(window)
        
#         # Handle NoData values (WorldPop typically uses -99999)
#         nodata = src.nodata
#         print(f"NoData value: {nodata}")
        
#         if nodata is not None:
#             # Create mask for valid data
#             valid_mask = hcmc_data != nodata
#             hcmc_population = np.sum(hcmc_data[valid_mask])
            
#             print(f"Pixels in HCMC area: {hcmc_data.shape}")
#             print(f"Valid pixels: {np.sum(valid_mask)}")
#             print(f"Ho Chi Minh City Population (2020): {hcmc_population:,.0f}")
            
#             # Additional statistics
#             if np.sum(valid_mask) > 0:
#                 density_data = hcmc_data[valid_mask]
#                 print(f"Average density: {np.mean(density_data):.2f} people/pixel")
#                 print(f"Maximum density: {np.max(density_data):.0f} people/pixel")
#                 print(f"Population range: {np.min(density_data):.0f} - {np.max(density_data):.0f}")
            
#             return hcmc_data, hcmc_transform, valid_mask
        
#         else:
#             hcmc_population = np.sum(hcmc_data)
#             print(f"Ho Chi Minh City Population (2020): {hcmc_population:,.0f}")
#             return hcmc_data, hcmc_transform, None

# # Get HCMC population
# hcmc_data, transform, mask = get_hcmc_population('./data/vnm_ppp_2020.tif')

# # Visualize the results
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

import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def get_hcmc_population_with_shapefile(tif_file, shapefile_path):
    # Read HCMC boundary
    hcmc_gdf = gpd.read_file(shapefile_path)
    print(f"HCMC boundary CRS: {hcmc_gdf.crs}")
    
    with rasterio.open(tif_file) as src:
        # Reproject HCMC boundary to match raster CRS if needed
        if hcmc_gdf.crs != src.crs:
            hcmc_gdf = hcmc_gdf.to_crs(src.crs)
        
        # Mask the raster with HCMC boundary
        hcmc_data, hcmc_transform = rasterio.mask.mask(
            src, 
            hcmc_gdf.geometry, 
            crop=True, 
            all_touched=True  # Include all pixels that touch the boundary
        )
        
        hcmc_data = hcmc_data[0]  # Get first band
        
        # Handle NoData values
        nodata = src.nodata
        print(f"NoData value: {nodata}")
        
        if nodata is not None:
            valid_mask = hcmc_data != nodata
            hcmc_population = np.sum(hcmc_data[valid_mask])
        else:
            hcmc_population = np.sum(hcmc_data)
            valid_mask = np.ones_like(hcmc_data, dtype=bool)
        
        print(f"Ho Chi Minh City Population (2020): {hcmc_population:,.0f}")
        print(f"Total area pixels: {hcmc_data.size}")
        print(f"Valid population pixels: {np.sum(valid_mask)}")
        
        return hcmc_data, hcmc_transform, valid_mask, hcmc_gdf

# If you have a HCMC shapefile, use this:
hcmc_data, transform, mask, hcmc_gdf = get_hcmc_population_with_shapefile(
    './data/vnm_ppp_2020.tif', 
    './shapefile/hochiminh.shp'
)