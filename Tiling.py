import slideio
import os
import numpy as np

OPENSLIDE_PATH = r'C:\Users\lasse\Documents\INSA\Projet_5BIM\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

path = "Data/Slides_cases_BAP1_test/03000195-00072192-1_1_1_HES.svs"
# slide = slideio.open_slide(path=path,driver="SVS")
slide = openslide.OpenSlide(path)
print(slide.properties)

tile_size = 512
threshold_RGB = 215
threshold_perc_white = 0.5 

#scene = slide.get_scene(0)
width,height = slide.dimensions
w_tile_number= width//tile_size
h_tile_number= height//tile_size
print(w_tile_number,h_tile_number)
for i in range(w_tile_number):
    for j in range(h_tile_number):
        tile = np.array(slide.read_region((i*tile_size,j*tile_size),0,(tile_size,tile_size)))
        if ((tile > threshold_RGB).sum(2) == 4).sum() / (tile.shape[0] * tile.shape[1]) < threshold_perc_white:
            np.save('C:/Users/lasse/Documents/INSA/Projet_5BIM/Tiles/'+path[40:45]+'_'+str(i)+'_'+str(j), tile)


