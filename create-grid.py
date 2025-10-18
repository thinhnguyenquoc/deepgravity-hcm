import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np

def create_osm_based_grid():
    # Get District 1 boundary (you might need to adjust the query)
    place_name = "District 1, Ho Chi Minh City, Vietnam"
    
    try:
        # Get boundary geometry
        boundary = ox.geocode_to_gdf(place_name)
        
        # Get the bounding box
        bbox = boundary.total_bounds  # [minx, miny, maxx, maxy]
        
        # Create grid within the boundary
        grid_cells = []
        cell_size_deg = 0.045/10  # Approximately 0.5km
        
        x_min, y_min, x_max, y_max = bbox
        
        x_current = x_min
        cell_id = 1
        
        while x_current < x_max:
            y_current = y_min
            while y_current < y_max:
                cell = Polygon([
                    (x_current, y_current),
                    (x_current + cell_size_deg, y_current),
                    (x_current + cell_size_deg, y_current + cell_size_deg),
                    (x_current, y_current + cell_size_deg)
                ])
                
                # Only include cells that intersect with District 1 boundary
                if cell.intersects(boundary.geometry.iloc[0]):
                    grid_cells.append({
                        'cell_id': cell_id,
                        'geometry': cell,
                        'center_lat': y_current + cell_size_deg/2,
                        'center_lon': x_current + cell_size_deg/2
                    })
                    cell_id += 1
                
                y_current += cell_size_deg
            x_current += cell_size_deg
        
        grid_gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, 12))
        boundary.plot(ax=ax, color='lightgray', alpha=0.5)
        grid_gdf.plot(ax=ax, alpha=0.3, edgecolor='red', facecolor='none')
        # Add cell IDs
        for idx, row in grid_gdf.iterrows():
            ax.annotate(
                text=str(row['cell_id']),
                xy=(row['center_lon'], row['center_lat']),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                color='darkred',
                bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none')
            )
        plt.title('0.5km Grid Over District 1, Ho Chi Minh City')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return grid_gdf
        
    except Exception as e:
        print(f"Error: {e}")
        print("Using fallback coordinates...")
        return create_precise_5km_grid()

# Create OSM-based grid
osm_grid = create_osm_based_grid()