import datetime
import osmnx as ox
import matplotlib.pyplot as plt

# Configure OSMnx
ox.settings.log_console = True
ox.settings.use_cache = True

# Get Ho Chi Minh City boundary
gdf = ox.geocode_to_gdf("Ho Chi Minh City, Vietnam")

# Plot the boundary
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)
ax.set_title("Ho Chi Minh City Boundary")
# plt.tight_layout()
# plt.show()

# Print boundary information
print(f"Boundary type: {gdf.geometry.iloc[0].geom_type}")
print(f"Area: {gdf.area.iloc[0]:.2f} square units")

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
output_dir = "./shapefile/hochiminh.shp"
gdf.to_file(output_dir, driver='ESRI Shapefile')

print("Shapefile exported with additional attributes!")
print(f"Area: {gdf['area_sqkm'].iloc[0]:.2f} sq km")