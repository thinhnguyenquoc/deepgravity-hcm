import osmnx as ox
import pandas as pd

def get_pois_osmnx(lat, lon, radius=1000, tags=None):
    """
    Get POIs using osmnx library (new version)
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius (int): Radius in meters
        tags (dict): OSM tags to search for
    """
    
    if tags is None:
        tags = {
            'amenity': True,
            'shop': True,
            'tourism': True,
            'leisure': True,
            'office': True
        }
    
    # Create point
    point = (lat, lon)
    
    try:
        # Get POIs within radius - NEW SYNTAX
        gdf = ox.features_from_point(point, tags=tags, dist=radius)
        
        # Convert to list of dictionaries
        pois = []
        for idx, row in gdf.iterrows():
            poi = {
                'id': idx,
                'geometry': row.geometry,
                'lat': row.geometry.centroid.y,
                'lon': row.geometry.centroid.x
            }
            
            # Add all tags as attributes
            for col in gdf.columns:
                if col != 'geometry' and pd.notna(row[col]):
                    poi[col] = row[col]
            
            pois.append(poi)
        
        print(f"Found {len(pois)} POIs")
        print(pois[:5])  # Print first 5 POIs for inspection
        return pois
    
    except Exception as e:
        print(f"Error: {e}")
        return []

# Example usage
pois = get_pois_osmnx(40.7589, -73.9851, radius=500)