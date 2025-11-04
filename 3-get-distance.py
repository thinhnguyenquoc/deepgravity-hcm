import json
import osmnx as ox
import networkx as nx
import asyncio

def get_driving_distance_specific_points(point1=(10.7730, 106.6984), point2=(10.8028, 106.7413)):
    # Known coordinates for central points in each district
    # District 1 center (around Ben Thanh Market)
    # point1 = 
    
    # District 2 center (around Thao Dien)
    # point2 = 
    
    try:
        # Create graph around Ho Chi Minh City
        G = ox.graph_from_point(point1, dist=10000, network_type='drive')
        
        # Get nearest nodes
        node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
        node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])
        
        # Calculate shortest path
        # route = nx.shortest_path(G, node1, node2, weight='length')
        
        # Calculate distance
        distance_meters = nx.shortest_path_length(G, node1, node2, weight='length')
        distance_km = distance_meters / 1000
        
        # print(f"Driving distance between specific points: {distance_km:.2f} km")
        
        # # Optional: Plot the route
        # fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, 
        #                             node_size=0, bgcolor='k', show=False, close=False)
        # ax.set_title(f"Route from District 1 to District 2: {distance_km:.2f} km")
        
        return distance_km
        
    except Exception as e:
        print(f"Error: {e}")
        return None

async def grid_distance(filename):
    try:
        # Specify the filename
        filename_grid = "./grid.json"
        data = []
        with open(filename_grid, 'r') as file:
            data = json.load(file)
        district_1 = None
        district_10 = None
        for item in data:
            if item["place_name"] == "district 1, Ho Chi Minh City, Vietnam":
                district_1 = item

            if item["place_name"] == "district 10, Ho Chi Minh City, Vietnam":
                district_10 = item

        matrix_distance = []
        matrix_distance_item = {}
        matrix_distance_item["tile_from"] = district_1["place_name"] + " to " + district_10["place_name"]
        matrix_distance_item["distances"] = []
        matrix_distance.append(matrix_distance_item)
        for cell1 in district_1["cells"]:
            for cell2 in district_10["cells"]:
                center1 = (cell1["center_lat"], cell1["center_lon"])
                center2 = (cell2["center_lat"], cell2["center_lon"])
                distance = get_driving_distance_specific_points(center1, center2)
                matrix_distance_item["distances"].append({
                    "cell_from": cell1["cell_id"],
                    "cell_to": cell2["cell_id"],
                    "distance_km": distance
                })
                print(f"Distance between cell {cell1['cell_id']} and cell {cell2['cell_id']}: {distance:.2f} km")
                with open(filename, 'w') as f:
                    json.dump(matrix_distance, f, indent=4) # 'indent=4' for pretty-printing
                await asyncio.sleep(2)
        
    except Exception as e:
        print(f"Error reading grid file: {e}")
        return None
# Run the function
# driving_distance = get_driving_distance_specific_points()
# print(f"Driving distance between District 1 and District 2: {driving_distance:.2f} km")
asyncio.run(grid_distance('./matrix_distance.json'))