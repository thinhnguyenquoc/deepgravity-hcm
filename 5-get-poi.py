import asyncio
import json
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
            'office': True,
            'public_transport': True
        }
    
    # Create point
    point = (lat, lon)
    
    try:
        # Get POIs within radius - NEW SYNTAX
        gdf = ox.features_from_point(point, tags=tags, dist=radius)
        
        # Convert to list of dictionaries
        pois = []
        pois_amenity = []
        pois_shop = []
        pois_tourism = []
        pois_leisure = []
        pois_office = []
        pois_public_transport = []

        for idx, row in gdf.iterrows():
            poi = {
                'id': idx,
                # 'geometry': row.geometry,
                'lat': row.geometry.centroid.y,
                'lon': row.geometry.centroid.x
            }
            
            # Add all tags as attributes
            for col in gdf.columns:
                if col != 'geometry' and pd.notna(row[col]):
                    poi[col] = row[col]
                if col == 'amenity' and pd.notna(row[col]):
                    pois_amenity.append(poi)
                if col == 'shop' and pd.notna(row[col]):
                    pois_shop.append(poi)
                if col == 'tourism' and pd.notna(row[col]):
                    pois_tourism.append(poi)
                if col == 'leisure' and pd.notna(row[col]):
                    pois_leisure.append(poi)
                if col == 'office' and pd.notna(row[col]):
                    pois_office.append(poi)
                if col == 'public_transport' and pd.notna(row[col]):
                    pois_public_transport.append(poi)

            pois.append(poi)
        
        # print(f"Found {len(pois)} POIs, including: amenity({len(pois_amenity)}), shop({len(pois_shop)}), tourism({len(pois_tourism)}), leisure({len(pois_leisure)}), office({len(pois_office)}), public_transport({len(pois_public_transport)})")
        # print(pois[:5])  # Print first 5 POIs for inspection
        return [pois, len(pois_amenity), len(pois_shop), len(pois_tourism), len(pois_leisure), len(pois_office), len(pois_public_transport)]
    
    except Exception as e:
        print(f"Error: {e}")
        return []

# Example usage
# pois = get_pois_osmnx(40.7589, -73.9851, radius=500)

async def grid_poi(district_name, filename, save=False):
    try:
        # Specify the filename
        filename_grid = "./grid.json"
        data = []
        with open(filename_grid, 'r') as file:
            data = json.load(file)
        district_from = None
        for item in data:
            if item["place_name"] == district_name:
                district_from = item
        
        with open(filename, 'r') as file:
            pois_file = json.load(file)
        poi_item = {}
        poi_item["place_name"] = district_from["place_name"]
        poi_item["cell_pois"] = []
        pois_file.append(poi_item)
        for cell1 in district_from["cells"]:
            pois_data = get_pois_osmnx(cell1["center"][1], cell1["center"][0], radius=500)
            if len(pois_data) > 0:
                pois, pois_amenity, pois_shop, pois_tourism, pois_leisure, pois_office, pois_public_transport = pois_data
                poi_item["cell_pois"].append({
                    "cell_id": cell1["cell_id"],
                    "pois": pois,
                    "pois_amenity": pois_amenity,
                    "pois_shop": pois_shop,
                    "pois_tourism": pois_tourism,
                    "pois_leisure": pois_leisure,
                    "pois_office": pois_office,
                    "pois_public_transport": pois_public_transport
                })
            else:
                poi_item["cell_pois"].append({
                    "cell_id": cell1["cell_id"],
                    "pois": [],
                    "pois_amenity": 0,
                    "pois_shop": 0,
                    "pois_tourism": 0,
                    "pois_leisure": 0,
                    "pois_office": 0,
                    "pois_public_transport": 0
                })
                print(f"Found no POIs in cell {cell1['cell_id']}")
                
            if save:
                with open(filename, 'w') as f:
                    json.dump(pois_file, f, indent=4) # 'indent=4' for pretty-printing
            await asyncio.sleep(5)
        
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None
    
# Run the function
asyncio.run(grid_poi('District 10', './pois_district.json', save=True))