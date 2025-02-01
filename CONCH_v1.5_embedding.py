import os
import numpy as np
import csv
import glob
import pandas as pd
import sys
from PIL import Image
import torch
import glob
import csv
import pandas as pd
from huggingface_hub import login
from transformers import AutoModel 

OPENSLIDE_PATH =r'C:\Users\CHALAS\source\repos\openslide-bin-4.0.0.6-windows-x64'
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def tiling_embedding (path, conch, eval_transform, tile_size, threshold_RGB, threshold_perc_white, bap1, wsi_id):
    with open('H:/PFAR/Mesopath_INSA/CONCH_embeddings/CONCH_embeddings_'+wsi_id+'.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        writer1.writerow(["ID", "Slide", "BAP1_mutation","Coord_x","Coord_y"]+ [f'Comp_{i}' for i in range(1, 769)])
        slide = openslide.OpenSlide(path)
        width,height = slide.dimensions
        w_tile_number= width//tile_size
        h_tile_number= height//tile_size
        for i in range(w_tile_number):
            for j in range(h_tile_number):
                tile = slide.read_region((i*tile_size,j*tile_size),0,(tile_size,tile_size)).convert("RGB")
                tile_np = np.array(tile)
                if ((tile_np > threshold_RGB).sum(2) == 3).sum() / (tile_np.shape[0] * tile_np.shape[1]) < threshold_perc_white:
                    print("Tile:",i,j)
                    image = eval_transform(tile).unsqueeze(0)
                    image = image.to(device)
                    with torch.inference_mode():
                        image_embs = conch(image)
                    writer1.writerow([wsi_id+'_'+str(i)+'_'+str(j), new_id, bap1, i*tile_size, j*tile_size]+image_embs[0].tolist())


### MAIN ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tile_size = 512
threshold_RGB = 220 # 215
threshold_perc_white = 0.5
tile_size = 512
threshold_RGB = 220 # 215
threshold_perc_white = 0.5
path_to_wsi = 'F:/MESO_AI/'

meta=pd.read_excel(r"C:\Users\CHALAS\source\repos\INSA_project_slides_anonym.xlsx", sheet_name="Feuil1", usecols="A,B")

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
conch, eval_transform = titan.return_conch()
conch = conch.to(device)

for i in range(0,len(meta.iloc[:,1])) :
        new_id="wsi_"+str(i).rjust(3, '0')
        print("Encoding slide:", meta.iloc[i,0])
        bap1 = meta.iloc[i,1]
        tiling_embedding(path_to_wsi+meta.iloc[i,0], conch, eval_transform, tile_size,threshold_RGB, threshold_perc_white, bap1, str(meta.iloc[i,0]))



