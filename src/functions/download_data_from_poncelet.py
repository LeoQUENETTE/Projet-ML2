import os, requests, zipfile, io
import pandas as pd

def download_data_from_poncelet(target_dir: str = "flickr_subset2") -> pd.DataFrame:
    return "test poncelet"
    url = "https://www.lirmm.fr/~poncelet/Ressources/flickr_subset2.zip"

    # # Check if data already exists
    # if not (os.path.exists(target_dir) and os.path.isdir(target_dir)):
    #     print("Téléchargement de flickr_subset2.zip...")
    #     response = requests.get(url)
    #     if response.status_code == 200:
    #         print("Téléchargement réussi. Extraction...")
    #         with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    #             # Extract without adding extra subfolder
    #             for member in zip_ref.namelist():
    #                 member_path = member
    #                 if member.startswith(target_dir+"/"):
    #                     member_path = member[len(target_dir+"/"):]
    #                 target_path = os.path.join(target_dir, member_path)

    #                 if member.endswith("/"):
    #                     os.makedirs(target_path, exist_ok=True)
    #                 else:
    #                     os.makedirs(os.path.dirname(target_path), exist_ok=True)
    #                     with zip_ref.open(member) as source, open(target_path, "wb") as target:
    #                         target.write(source.read())
    #         print(f"Données extraites dans : {target_dir}")
    #     else:
    #         print("Échec du téléchargement. Code HTTP :", response.status_code)
    #         return None
    # else:
    #     print("Données déjà disponibles dans :", target_dir)
    
    # # Read and return the CSV
    # csv_path = os.path.join(target_dir, "captions.csv")
    # return pd.read_csv(csv_path)