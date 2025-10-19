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
plt.tight_layout()
plt.show()

# Print boundary information
print(f"Boundary type: {gdf.geometry.iloc[0].geom_type}")
print(f"Area: {gdf.area.iloc[0]:.2f} square units")