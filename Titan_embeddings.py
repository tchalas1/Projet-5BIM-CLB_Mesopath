from huggingface_hub import login
from transformers import AutoModel 
import torch 
import pandas as pd
import os


def titan_embbed(file_path):
    # Load the CSV
    conch_embbed = pd.read_csv(file_path, sep=",")
    
    # Extract coordinates and features
    coords = conch_embbed[['Coord_x', 'Coord_y']].values
    features = conch_embbed.loc[:, conch_embbed.columns.str.startswith('Comp_')].values
    
    # Convert to tensors
    coords_tensor = torch.tensor(coords, dtype=torch.int64).unsqueeze(0)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    # Encode the slide and save the embedding
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        features_tensor = features_tensor.to(device)
        coords_tensor = coords_tensor.to(device)
        slide_embedding = titan.encode_slide_from_patch_features(features_tensor, coords_tensor, patch_size_lv0)
        
        file_name=file_path.split("/")
        
        # Save the embedding with a meaningful name
        output_path = file_name[-1].replace("Conch", "Titan") 
        output_path = output_path.replace(".svs.csv", ".pt") # Change extension to .pt
        print(output_path)
        torch.save(slide_embedding, save_path + output_path)
        print(f"Embedding saved to {output_path}")


login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
conch, eval_transform = titan.return_conch()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
titan = titan.to(device)
patch_size_lv0 = 512 #kepts as in the paper for x20 magnification

path_to_conch_embedd="C:/Users/thiba/Documents/INSA_2024-2025/Semestre1/projet_5BIM/Conch_Embbedings/"
save_path = "C:/Users/thiba/Documents/INSA_2024-2025/Semestre1/projet_5BIM/Titan_embeddings/"

# Iterate through the files
for file_name in os.listdir(path_to_conch_embedd):
    file_path = os.path.join(path_to_conch_embedd, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".csv"):  # Process only .csv files
        try:
            titan_embbed(file_path)
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")