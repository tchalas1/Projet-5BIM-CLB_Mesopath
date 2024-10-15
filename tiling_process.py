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

# Tiling (assuming 512x512 pixels)
x, y = 0, 0  # Top-left corner
tile_size = 512
tile = slide.read_region((x, y), 0, (tile_size, tile_size))

# Convert to numpy for further processing
tile_np = np.array(tile)
Image.fromarray(tile_np).show()  # Show tile
