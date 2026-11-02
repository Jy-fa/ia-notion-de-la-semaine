"""Vérifie que l'environnement est prêt pour le module Embeddings et bases vectorielles.

Le premier lancement télécharge le modèle all-MiniLM-L6-v2 (environ 90 Mo).
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("1. Bibliothèques importées")
print(f"   NumPy {np.__version__}, FAISS {faiss.__version__}")

# Transformer trois phrases en vecteurs
model = SentenceTransformer("all-MiniLM-L6-v2")
phrases = [
    "Le chat dort sur le canapé.",
    "Le félin se repose sur le sofa.",
    "Le projet Flink est à rendre fin novembre.",
]
vecteurs = model.encode(phrases, normalize_embeddings=True)

print("\n2. Embeddings calculés")
print(f"   Forme de la matrice : {vecteurs.shape}  (3 phrases × {vecteurs.shape[1]} dimensions)")

# Similarité cosinus : sur des vecteurs normalisés, c'est un simple produit scalaire
sim_proches = float(vecteurs[0] @ vecteurs[1])
sim_eloignees = float(vecteurs[0] @ vecteurs[2])
print("\n3. Similarité cosinus")
print(f"   « chat / canapé » vs « félin / sofa » : {sim_proches:.2f}")
print(f"   « chat / canapé » vs « projet Flink »  : {sim_eloignees:.2f}")

# Une première recherche avec FAISS
index = faiss.IndexFlatIP(vecteurs.shape[1])
index.add(vecteurs)
scores, ids = index.search(vecteurs[:1], k=3)
print("\n4. Recherche FAISS pour la phrase 1")
for rang, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
    print(f"   {rang}. {phrases[i]}  (score {s:.2f})")

print("\nTout fonctionne. L'environnement est prêt.")
