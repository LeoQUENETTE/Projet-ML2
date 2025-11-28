# Notes sur le projet

## CLIP

1) On reprend les modèles précédents, on dégage les couches de classification pour garder que les embeddings
   1) On récupère les modèles enregistrés et on suprime les dernière couches, on veut éviter d'avoir à réentrainé le modèle
2) On utilise ensuite les deux modèles de création d'embeddings précédent pour avoir un modèle CLIP
   1) Utilisation de loss_contrastive (code donné)
   2) Important de sauvegarder le modèle
3) Faites de l'inférence, test à la mano