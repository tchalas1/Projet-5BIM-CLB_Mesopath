import os
import numpy as np
import csv
import glob
import pandas as pd
import sys
sys.path.insert(0, '/Users/CHALAS/')
from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
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


def tiling_embedding (path, model, preprocess, tile_size, threshold_RGB, threshold_perc_white, bap1, wsi_id):
    with open(path_to_embeds + 'CONCH_embeddings_'+wsi_id+'.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        writer1.writerow(["ID", "Slide", "BAP1_mutation","Coord_x","Coord_y"]+ [f'Comp_{i}' for i in range(1, 513)])
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
                    image = preprocess(tile).unsqueeze(0)
                    image = image.to(device)
                    with torch.inference_mode():
                        image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
                    writer1.writerow([wsi_id+'_'+str(i)+'_'+str(j), new_id, bap1, i*tile_size, j*tile_size]+image_embs[0].tolist())


### MAIN ###

tile_size = 512
threshold_RGB = 220 # 215
threshold_perc_white = 0.5
tile_size = 512
threshold_RGB = 220 # 215
threshold_perc_white = 0.5

# Adapter les chemins !
path_to_wsi = 'F:/MESO_AI/'
path_to_meta = r"C:\Users\CHALAS\source\repos\INSA_project_slides_anonym.xlsx"
path_to_embeds = 'H:/PFAR/Mesopath_INSA/CONCH_embeddings/' # Chemin vers le dossier ou doivent être enregistrer les embeddings
path_to_conch_ckpt = r"C:\Users\CHALAS\source\repos\CONCH\checkpoints\conch\pytorch_model.bin"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta=pd.read_excel(path_to_meta, sheet_name="Feuil1", usecols="A,B")
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=path_to_conch_ckpt , force_image_size=512)
model = model.to(device)

for i in range(0,len(meta.iloc[:,1])) :
        new_id="wsi_"+str(i).rjust(3, '0')
        print("Encoding slide:", meta.iloc[i,0])
        bap1 = meta.iloc[i,1]
        tiling_embedding(path_to_wsi+meta.iloc[i,0], model,preprocess, tile_size,threshold_RGB, threshold_perc_white, bap1, str(meta.iloc[i,0]))



