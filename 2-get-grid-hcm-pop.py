# from shapely.geometry import box
import json
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
def extract_hcmc_population(district_name, raster_path, hcmc_boundary_path=None, is_save=False):
    """
    Extract population data for Ho Chi Minh City
    """
    # If you have a shapefile for HCMC boundaries
    if hcmc_boundary_path:
        filename = "./grid.json"
        with open(filename, 'r') as file:
            data_grid = json.load(file)
        for item in data_grid:
            if item["place_name"] == district_name:
                print("have existed " + district_name)
                return
        grid = {'place_name': district_name, 'cells': []}
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
                    x2, y2 = out_transform * (col + 1, row + 1)
                    x3, y3 = out_transform * (col, row + 1)
                    x4, y4 = out_transform * (col + 0.5, row + 0.5)

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
                        lon4, lat4 = transform(src.crs, 'EPSG:4326', [x4], [y4])
                        lon4, lat4 = lon4[0], lat4[0]
                    else:
                        lon, lat = x, y
                        lon1, lat1 = x1, y1
                        lon2, lat2 = x2, y2
                        lon3, lat3 = x3, y3
                        lon4, lat4 = x4, y4
                    
                    # print(f"Cell {cell_count+1}: Row={row}, Col={col}, "
                    #     f"Lat={lat:.6f}, Lon={lon:.6f}, Value={value}")
                    print(f"{lon:.6f}, {lat:.6f}")
                    print(f"{lon1:.6f}, {lat1:.6f}")
                    print(f"{lon2:.6f}, {lat2:.6f}")
                    print(f"{lon3:.6f}, {lat3:.6f}")

                    item = {}
                    item["cell_id"] = cell_count
                    item["polygon"] = [
                        (lon, lat),
                        (lon1, lat1),
                        (lon2, lat2),
                        (lon3, lat3)
                    ]
                    item['center'] = (lon4, lat4)
                    item['pop'] = int(value)
                    grid['cells'].append(item)
                    # save position (lat, lon, value) to list or dictionary as needed
                    cell_count += 1

            print(cell_count)
            data_grid.append(grid)
            # Open the file in write mode ('w') and use json.dump() to write the data
            if is_save:
                with open(filename, 'w') as f:
                    json.dump(data_grid, f, indent=4) # 'indent=4' for pretty-printing
            hcmc_population = out_image[0]
    
    return hcmc_population

# Extract HCMC population
district_name = "District 7"
hcmc_pop = extract_hcmc_population(district_name, "./data/vnm_ppp_2020_1km_Aggregated.tif",'./shapefile/district_7_hcmc.shp', is_save=False)

# Plot HCMC population
plt.figure(figsize=(8, 6))
plt.imshow(hcmc_pop, cmap='YlOrRd')
plt.colorbar(label='Population per 1km²')
plt.title(f'{district_name} - population Distribution 2020')
plt.axis('off')
plt.show()
