# from shapely.geometry import box
import geopandas as gpd
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sympy import false

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    # print(element)
    if element is None: 
        return False
    try:
        a =float(element)
        if np.isnan(a):
            return False
        return True
    except ValueError:
        return False
def extract_hcmc_population(raster_path, hcmc_boundary_path=None):
    """
    Extract population data for Ho Chi Minh City
    """
    # If you have a shapefile for HCMC boundaries
    if hcmc_boundary_path:
        # Load HCMC boundary
        hcmc_gdf = gpd.read_file(hcmc_boundary_path)
        
        with rasterio.open(raster_path) as src:
            # Mask the raster with HCMC boundary
            out_image, out_transform = rasterio.mask.mask(
                src, 
                hcmc_gdf.geometry, 
                crop=True, 
                filled=False
            )

             # Get data and coordinates
            data = out_image[0]  # First band
            height, width = data.shape
            max_cells = height * width
            # Print cell information
            print(f"\nFirst {max_cells} cells:")
            print("-" * 70)
            
            cell_count = 0
            for row in range(height):
                for col in range(width):
                    if cell_count >= max_cells:
                        break
                        
                    value = data[row, col]
                    
                    if is_float(value) == False:
                        # print("no data", value)
                        continue
                    x, y = out_transform * (col, row)
                    x1, y1 = out_transform * (col + 1, row)
                    x2, y2 = out_transform * (col, row + 1)
                    x3, y3 = out_transform * (col + 1, row + 1)

                    # Transform coordinates
                    if src.crs and not src.crs.is_geographic:
                        from rasterio.warp import transform
                        lon, lat = transform(src.crs, 'EPSG:4326', [x], [y])
                        lon, lat = lon[0], lat[0]
                        lon1, lat1 = transform(src.crs, 'EPSG:4326', [x1], [y1])
                        lon1, lat1 = lon1[0], lat1[0]
                        lon2, lat2 = transform(src.crs, 'EPSG:4326', [x2], [y2])
                        lon2, lat2 = lon2[0], lat2[0]
                        lon3, lat3 = transform(src.crs, 'EPSG:4326', [x3], [y3])
                        lon3, lat3 = lon3[0], lat3[0]
                    else:
                        lon, lat = x, y
                        lon1, lat1 = x1, y1
                        lon2, lat2 = x2, y2
                        lon3, lat3 = x3, y3
                    
                    # print(f"Cell {cell_count+1}: Row={row}, Col={col}, "
                    #     f"Lat={lat:.6f}, Lon={lon:.6f}, Value={value}")
                    print(f"{lon:.6f}, {lat:.6f}")
                    print(f"{lon1:.6f}, {lat1:.6f}")
                    print(f"{lon2:.6f}, {lat2:.6f}")
                    print(f"{lon3:.6f}, {lat3:.6f}")

                    # save position (lat, lon, value) to list or dictionary as needed
                    cell_count += 1

                # print(cell_count)
                hcmc_population = out_image[0]
    
    return hcmc_population

# Extract HCMC population
district_name = "District 1"
hcmc_pop = extract_hcmc_population("./data/vnm_ppp_2020_1km_Aggregated.tif",'./shapefile/district1_hcmc.shp')

# Plot HCMC population
plt.figure(figsize=(8, 6))
plt.imshow(hcmc_pop, cmap='YlOrRd')
plt.colorbar(label='Population per 1km²')
plt.title(f'{district_name} - population Distribution 2020')
plt.axis('off')
plt.show()
