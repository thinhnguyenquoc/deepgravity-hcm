import osmnx as ox
import networkx as nx

def get_driving_distance_specific_points():
    # Known coordinates for central points in each district
    # District 1 center (around Ben Thanh Market)
    point1 = (10.7730, 106.6984)
    
    # District 2 center (around Thao Dien)
    point2 = (10.8028, 106.7413)
    
    try:
        # Create graph around Ho Chi Minh City
        G = ox.graph_from_point(point1, dist=10000, network_type='drive')
        
        # Get nearest nodes
        node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
        node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])
        
        # Calculate shortest path
        route = nx.shortest_path(G, node1, node2, weight='length')
        
        # Calculate distance
        distance_meters = nx.shortest_path_length(G, node1, node2, weight='length')
        distance_km = distance_meters / 1000
        
        print(f"Driving distance between specific points: {distance_km:.2f} km")
        
        # Optional: Plot the route
        fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, 
                                    node_size=0, bgcolor='k', show=False, close=False)
        ax.set_title(f"Route from District 1 to District 2: {distance_km:.2f} km")
        
        return distance_km
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run the function
driving_distance = get_driving_distance_specific_points()