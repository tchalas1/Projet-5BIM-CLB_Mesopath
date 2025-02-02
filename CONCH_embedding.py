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


def tiling_embedding(path, model, preprocess, tile_size, threshold_RGB, threshold_perc_white, bap1, wsi_id):
    """
    Fonction qui segmente une image WSI en tuiles, filtre les tuiles trop blanches,
    et extrait les embeddings à l'aide d'un modèle de deep learning.
    """
    with open(path_to_embeds + 'CONCH_embeddings_' + wsi_id + '.csv', 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        # Écriture de l'en-tête du fichier CSV
        writer1.writerow(["ID", "Slide", "BAP1_mutation", "Coord_x", "Coord_y"] + [f'Comp_{i}' for i in range(1, 513)])
        
        # Chargement de la WSI
        slide = openslide.OpenSlide(path)
        width, height = slide.dimensions
        w_tile_number = width // tile_size
        h_tile_number = height // tile_size
        
        # Parcours de l'image par tuiles
        for i in range(w_tile_number):
            for j in range(h_tile_number):
                tile = slide.read_region((i * tile_size, j * tile_size), 0, (tile_size, tile_size)).convert("RGB")
                tile_np = np.array(tile)
                
                # Vérification du pourcentage de pixels blancs
                if ((tile_np > threshold_RGB).sum(2) == 3).sum() / (tile_np.shape[0] * tile_np.shape[1]) < threshold_perc_white:
                    print("Tile:", i, j)
                    image = preprocess(tile).unsqueeze(0).to(device)
                    
                    # Extraction des embeddings avec le modèle
                    with torch.inference_mode():
                        image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
                    
                    # Écriture des résultats dans le CSV
                    writer1.writerow([wsi_id + '_' + str(i) + '_' + str(j), new_id, bap1, i * tile_size, j * tile_size] + image_embs[0].tolist())

### MAIN ###

# Paramètres de segmentation et de filtrage
tile_size = 512
threshold_RGB = 220  # Seuil pour détecter les pixels blancs
threshold_perc_white = 0.5  # Pourcentage max de pixels blancs autorisé par tuile

# Définition des chemins (à adapter selon l'environnement)
path_to_wsi = 'F:/MESO_AI/'  # Dossier contenant les WSIs
path_to_meta = r"C:\Users\CHALAS\source\repos\INSA_project_slides_anonym.xlsx"
path_to_embeds = 'H:/PFAR/Mesopath_INSA/CONCH_embeddings/'  # Dossier de sortie pour les embeddings
path_to_conch_ckpt = r"C:\Users\CHALAS\source\repos\CONCH\checkpoints\conch\pytorch_model.bin"

# Détection du périphérique pour PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des métadonnées
meta = pd.read_excel(path_to_meta, sheet_name="Feuil1", usecols="A,B")

# Chargement du modèle et du préprocesseur
model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=path_to_conch_ckpt, force_image_size=512)
model = model.to(device)

# Boucle de traitement des images WSIs
for i in range(len(meta.iloc[:, 1])):
    new_id = "wsi_" + str(i).rjust(3, '0')  # Génération d'un ID unique
    print("Encoding slide:", meta.iloc[i, 0])
    bap1 = meta.iloc[i, 1]  # Récupération de l'annotation BAP1
    
    # Appel de la fonction de segmentation et d'extraction des embeddings
    tiling_embedding(path_to_wsi + meta.iloc[i, 0], model, preprocess, tile_size, threshold_RGB, threshold_perc_white, bap1, str(meta.iloc[i, 0]))
