import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import numpy as np
from pyproj import Transformer

def create_osmnx_grid_with_boundaries():
    # Configure OSMnx
    ox.settings.log_console = True
    ox.settings.use_cache = True
    
    # Define District 1, Ho Chi Minh City
    place_name = "District 1, Ho Chi Minh City, Vietnam"
    
    try:
        # Get District 1 boundary
        print("Downloading District 1 boundary...")
        boundary_gdf = ox.geocode_to_gdf(place_name)
        
        # Get the bounding box
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
        print(f"District 1 bounds: {minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}")
        
        # Set up coordinate transformation for accurate 5km grid
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        
        # Convert bounds to meters for accurate grid
        minx_m, miny_m = transformer.transform(minx, miny)
        maxx_m, maxy_m = transformer.transform(maxx, maxy)

        print(minx_m, miny_m, maxx_m, maxy_m)
        
        # Create 5km grid in meters
        grid_size = 1000  # 5km in meters
        grid_cells = []
        cell_data = []
        
        x_current = minx_m
        cell_id = 1
        
        while x_current < maxx_m:
            y_current = miny_m
            while y_current < maxy_m:
                # Create cell in meter coordinates
                cell_meters = Polygon([
                    (x_current, y_current),
                    (x_current + grid_size, y_current),
                    (x_current + grid_size, y_current + grid_size),
                    (x_current, y_current + grid_size)
                ])
                
                # Convert back to geographic coordinates
                coords_geo = []
                for x, y in cell_meters.exterior.coords:
                    lon, lat = transformer_back.transform(x, y)
                    coords_geo.append((lon, lat))
                
                cell_geo = Polygon(coords_geo)
                
                # Check if cell intersects with District 1 boundary
                if cell_geo.intersects(boundary_gdf.geometry.iloc[0]):
                    # Calculate center point
                    center_x = x_current + grid_size / 2
                    center_y = y_current + grid_size / 2
                    center_lon, center_lat = transformer_back.transform(center_x, center_y)
                    
                    grid_cells.append(cell_geo)
                    cell_data.append({
                        'cell_id': cell_id,
                        'area_km2': 25,
                        'center_lat': center_lat,
                        'center_lon': center_lon
                    })
                    
                    cell_id += 1
                
                y_current += grid_size
            x_current += grid_size
        
        print("cell_data:", cell_data)
        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(cell_data, geometry=grid_cells, crs="EPSG:4326")
        
        print(f"Created {len(grid_gdf)} grid cells covering District 1")
        
        return grid_gdf, boundary_gdf
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Create the grid
grid_gdf, boundary_gdf = create_osmnx_grid_with_boundaries()

if grid_gdf is not None:
    # Plot with clear boundaries
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot District 1 boundary
    boundary_gdf.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='black', linewidth=2)
    
    # Plot grid cells with clear boundaries
    grid_gdf.plot(
        ax=ax, 
        alpha=0.2,           # Light fill
        edgecolor='red',     # Red boundaries
        linewidth=2,         # Thick boundaries
        facecolor='none'     # Transparent fill to see boundaries clearly
    )
    
    # Add cell IDs
    print("Annotating cell IDs on the grid...")
    
    for idx, row in grid_gdf.iterrows():
        ax.annotate(
            text=str(row['cell_id']),
            xy=(row['center_lon'], row['center_lat']),
            xytext=(0, 0),
            textcoords="offset points",
            ha='center',
            va='center',
            fontsize=8,
            fontweight='bold',
            color='darkred'
        )
    
    ax.set_title('1km × 1km Grid with Cell Boundaries - District 1, Ho Chi Minh City', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()