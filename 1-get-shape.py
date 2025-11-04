import datetime
import osmnx as ox
import matplotlib.pyplot as plt

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True

def plot_district_boundary(district_name, is_save = False):
    # Get the boundary of the specified district
    gdf = ox.geocode_to_gdf(f"{district_name}, Ho Chi Minh City, Vietnam")
    
    # Plot the boundary
    fig, ax = plt.subplots(figsize=(7, 7))
    gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
    ax.set_title(f"{district_name} Boundary")
   
    # Print boundary information
    print(f"Boundary type: {gdf.geometry.iloc[0].geom_type}")

    # Add additional attributes
    gdf['city_name'] = 'Ho Chi Minh City'
    gdf['country'] = 'Vietnam'
    gdf['area_sqkm'] = gdf.geometry.area / 10**6  # Convert to sq km
    gdf['download_date'] = datetime.datetime.now().strftime('%Y-%m-%d')

    # Reproject to a projected coordinate system (optional)
    # Common CRS for Vietnam: EPSG:4759 (Vietnam 2000)
    gdf = gdf.to_crs('EPSG:4759')

    # Recalculate area in proper projection
    gdf['area_sqkm'] = gdf.geometry.area / 10**6

    # Export to Shapefile
    output_dir = f"./shapefile/{str.strip(district_name).replace(' ', '_')}_hcmc.shp"
    if is_save:
        gdf.to_file(output_dir, driver='ESRI Shapefile')

    print("Shapefile exported with additional attributes!")
    plt.show()

    return gdf

plot_district_boundary("District 1")    
