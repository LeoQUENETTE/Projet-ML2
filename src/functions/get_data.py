import os, requests, zipfile, io
def get_data():
    url = "https://www.lirmm.fr/~poncelet/Ressources/flickr_subset2.zip"
    target_dir = "flickr_subset2"

    # Vérifie si le dossier existe déjà
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        print("Données déjà disponibles dans :", target_dir)
    else:
        print("Téléchargement de flickr_subset2.zip...")
        response = requests.get(url)
        if response.status_code == 200:
            print("Téléchargement réussi. Extraction...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                # Extraire sans ajouter de sous-dossier supplémentaire
                for member in zip_ref.namelist():
                    # Corrige les chemins pour ignorer un éventuel prefixe flickr_subset2/
                    member_path = member
                    if member.startswith("flickr_subset2/"):
                        member_path = member[len("flickr_subset2/"):]
                    target_path = os.path.join(target_dir, member_path)

                    # Si c'est un répertoire, on le crée
                    if member.endswith("/"):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            target.write(source.read())
            print(f"Données extraites dans : {target_dir}")
        else:
            print("Échec du téléchargement. Code HTTP :", response.status_code)