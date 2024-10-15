# Example: Loading and tiling using OpenSlide
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\Users\thiba\Documents\INSA_2024-2025\Semestre1\projet_5BIM\openslide-bin-4.0.0.6-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from PIL import Image
import numpy as np

# Load .svs file
slide = openslide.OpenSlide('C:/Users/thiba/Documents/INSA_2024-2025/Semestre1/projet_5BIM/transfer_8262829_files_4fcee2db/Slides_cases_BAP1_test/03000195-00072192-1_1_1_HES.svs')

# Get the dimensions of the whole-slide image
slide_width, slide_height = slide.dimensions

# Tile size
tile_size = 512

# List to store the tiles
tiles = []

# Loop over the image and extract non-overlapping tiles
for x in range(0, slide_width, tile_size):
    for y in range(0, slide_height, tile_size):
        # Read a region of the slide (a tile)
        tile = slide.read_region((x, y), 0, (tile_size, tile_size))
        tile = tile.convert("RGB")  # Convert to RGB
        tiles.append(np.array(tile))  # Store tile as numpy array

# Convert the list of tiles into a single numpy array
tiles_array = np.array(tiles)

# Save the numpy array to a file
np.save("tiles_array.npy", tiles_array)

print("Tiles array saved as 'tiles_array.npy'")