#import slideio
import os
import numpy as np
import torch
from torchvision import transforms
import torchstain

OPENSLIDE_PATH = r'C:\Users\lasse\Documents\INSA\Projet_5BIM\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def tiling (path, tile_size, threshold_RGB, threshold_perc_white):
    slide = openslide.OpenSlide(path)
    print(slide.properties)
    width,height = slide.dimensions
    w_tile_number= width//tile_size
    h_tile_number= height//tile_size
    print(w_tile_number,h_tile_number)
    # tile = slide.read_region((0,0),0,(tile_size,tile_size)).convert("RGB")
    # tile.save('C:/Users/lasse/Documents/INSA/Projet_5BIM/test.jpeg')
    # for i in range(w_tile_number):
    #     for j in range(h_tile_number):
    #         tile = slide.read_region((i*tile_size,j*tile_size),0,(tile_size,tile_size)).convert("RGB")
    #         tile_np = np.array(tile)
    #         if ((tile_np > threshold_RGB).sum(2) == 3).sum() / (tile_np.shape[0] * tile_np.shape[1]) < threshold_perc_white:
    #             tile.save('C:/Users/lasse/Documents/INSA/Projet_5BIM/Tiles/'+path[40:45]+'_'+str(i)+'_'+str(j)+'.jpeg')

tile_size = 512
threshold_RGB = 215
threshold_perc_white = 0.5 

tiling("Data/Slides_cases_BAP1_test/03000195-00072192-1_1_1_HES.svs",tile_size,threshold_RGB, threshold_perc_white)
