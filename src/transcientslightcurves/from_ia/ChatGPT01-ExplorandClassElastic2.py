#  


# # Exploration et mini-classification ELAsTiCC2
# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : 2025-11-06 at NERSC (**kernel desc-td-env-dev**)
# - last update : 2025-11-10 laptop on (**kernel pytorch-cpu-py312**) @

# ============================================================
# Notebook: Exploration et mini-classification ELAsTiCC2
# Kernel: desc-td-env-dev
# ============================================================

# -----------------------------
# 1️⃣ Imports
# -----------------------------



import os
import socket
import random
import numpy as np
import pandas as pd
import fitsio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



np.random.seed(42)




def is_on_nersc():
    """Détecte si le code tourne sur un système NERSC."""
    # Vérifier le nom d'hôte
    hostname = socket.gethostname()
    nersc_hostnames = ["perlmutter", "cori", "nersc.gov"]
    host_check = any(nersc_host in hostname for nersc_host in nersc_hostnames)

    # Vérifier les variables d'environnement
    nersc_vars = ["NERSC_HOST", "SLURM_JOB_ID", "CRAY_SYSTEM_NAME"]
    env_check = any(var in os.environ for var in nersc_vars)

    return host_check or env_check

if is_on_nersc():
    print("Je suis sur un système NERSC.")
else:
    print("Je suis sur mon laptop ou un autre système.")


# -----------------------------
# 2️⃣ Définir les chemins NERSC
# -----------------------------


if is_on_nersc():
    print("Configuration pour NERSC : utilisation des GPU ou des chemins spécifiques.")
    # Exemple : charger des données depuis un chemin NERSC
    BASE_PATH = "/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2"
else:
    print("Configuration pour laptop : utilisation des ressources locales.")
    # Exemple : charger des données depuis un chemin local
    BASE_PATH = "/Users/dagoret/DATA/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2"


# Liste des types de SN (sous-dossiers)
types_sn = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
print("Types disponibles :", types_sn[:10])




sample_types = [
    "ELASTICC2_TRAIN_02_SNIa-SALT3",
    "ELASTICC2_TRAIN_02_SNIc-Templates",
    "ELASTICC2_TRAIN_02_SNIb-Templates"
]




type_dir = os.path.join(BASE_PATH, "ELASTICC2_TRAIN_02_SNIa-SALT3")

# Lister HEAD et PHOT compressés
head_files = sorted([f for f in os.listdir(type_dir) if "HEAD" in f])
phot_files = sorted([f for f in os.listdir(type_dir) if "PHOT" in f])

# Choisir un objet pour test
head_file = head_files[0]
phot_file = phot_files[0]

head_data = fitsio.FITS(os.path.join(type_dir, head_file))[1].read()
phot_data = fitsio.FITS(os.path.join(type_dir, phot_file))[1].read()

print("HEAD colonnes :", head_data.dtype.names)
print("PHOT colonnes :", phot_data.dtype.names)

# Visualisation multibande
bands = set(phot_data["BAND"])
bands = {b for b in bands if b.strip() != '-'}
print(bands)

plt.figure(figsize=(8,5))
for b in bands:
    mask = phot_data["BAND"] == b
    plt.errorbar(phot_data["MJD"][mask], phot_data["FLUXCAL"][mask],
                 yerr=phot_data["FLUXCALERR"][mask], fmt='o', label=f'Band {b}')
plt.xlabel("MJD")
plt.ylabel("Flux")
plt.title("Courbe lumière exemple")
plt.legend()
plt.show()


# ✅ Ce que fait ce code :
# - Parcourt plusieurs types de SN sélectionnés.
# - Lit HEAD + PHOT compressés .FITS.gz.
# - Filtre les bandes bidon '-' et supprime les espaces.
# - Extrait pour chaque bande :
#     - Flux maximum
#     - MJD du pic
# - Extrait quelques features globales depuis HEAD (nombre d’observations NOBS, redshift final).
# - Retourne un DataFrame prêt pour un modèle ML et un vecteur de labels.


features_list = []
labels = []

for sn_type in sample_types:
    type_dir = os.path.join(BASE_PATH, sn_type)

    # Lister HEAD et PHOT
    head_files = sorted([f for f in os.listdir(type_dir) if "HEAD" in f])
    phot_files = sorted([f for f in os.listdir(type_dir) if "PHOT" in f])

    # Pour chaque objet, extraire features simples
    for head_file, phot_file in zip(head_files, phot_files):
        head_data = fitsio.FITS(os.path.join(type_dir, head_file))[1].read()
        phot_data = fitsio.FITS(os.path.join(type_dir, phot_file))[1].read()

        # Nettoyage des bandes
        bands = set(b.strip() for b in phot_data["BAND"] if b.strip() != '-')

        feature_dict = {}
        for b in bands:
            mask = np.array([x.strip() for x in phot_data["BAND"]]) == b
            if np.sum(mask) > 0:
                feature_dict[f"flux_max_{b}"] = np.max(phot_data["FLUXCAL"][mask])
                feature_dict[f"mjd_peak_{b}"] = phot_data["MJD"][mask][np.argmax(phot_data["FLUXCAL"][mask])]
            else:
                feature_dict[f"flux_max_{b}"] = 0
                feature_dict[f"mjd_peak_{b}"] = 0

        # Ajouter des features globales depuis HEAD si souhaité
        feature_dict["NOBS"] = head_data["NOBS"][0]
        feature_dict["REDSHIFT_FINAL"] = head_data["REDSHIFT_FINAL"][0]

        features_list.append(feature_dict)
        labels.append(sn_type)

# Convertir en DataFrame pour ML
X = pd.DataFrame(features_list)
y = np.array(labels)

print("Shape features :", X.shape)
print("Exemple features :\n", X.head())


# 🔹 Ce que fait ce code
# - Split train/test avec 70/30, stratifié par type de SN.
# - Entraîne un RandomForest sur les features extraites (flux_max par bande, MJD du pic, NOBS, REDSHIFT_FINAL).
# - Affiche un rapport de classification et une matrice de confusion.



# Séparer train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Créer le modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prédictions
y_pred = clf.predict(X_test)

# Évaluation
print("Classification report :\n")
print(classification_report(y_test, y_pred))

print("Confusion matrix :\n")
print(confusion_matrix(y_test, y_pred))




# Inférence sur la totalité des data

if not is_on_nersc():
    assert False

# -----------------------------
# 2️⃣ Définir les chemins NERSC
# -----------------------------
ELASTIC2_PATH = "/global/cfs/cdirs/desc-td/ELASTICC2/"
HEAD_PATH = os.path.join(ELASTIC2_PATH, "HEAD")
PHOT_PATH = os.path.join(ELASTIC2_PATH, "PHOT")



# Vérifier qu'il y a des fichiers
head_files = os.listdir(HEAD_PATH)
phot_files = os.listdir(PHOT_PATH)
print(f"HEAD files: {head_files[:5]}")
print(f"PHOT files: {phot_files[:5]}")

# -----------------------------
# 3️⃣ Lire un HEAD aléatoire
# -----------------------------
head_file = os.path.join(HEAD_PATH, random.choice(head_files))
head_data = fitsio.FITS(head_file)[1].read()
print("Colonnes HEAD :", head_data.dtype.names)
print("Exemple HEAD :\n", head_data[:5])

# -----------------------------
# 4️⃣ Lire un PHOT correspondant
# -----------------------------
# On prend l'objet SNID du HEAD
snid = head_data["SNID"][0]
phot_file = os.path.join(PHOT_PATH, f"PHOT-{head_file.split('-')[1]}.FITS")
phot_data = fitsio.FITS(phot_file)[1].read()
# Filtrer pour le SNID choisi
phot_sn = phot_data[phot_data["SNID"] == snid]
print("Colonnes PHOT :", phot_data.dtype.names)
print("Exemple PHOT :\n", phot_sn[:5])

# -----------------------------
# 5️⃣ Visualiser la courbe multibande
# -----------------------------
bands = np.unique(phot_sn["BAND"])
plt.figure(figsize=(8,5))
for b in bands:
    mask = phot_sn["BAND"] == b
    plt.errorbar(phot_sn["MJD"][mask], phot_sn["FLUXCAL"][mask],
                 yerr=phot_sn["FLUXCALERR"][mask], fmt='o', label=f'Band {b}')
plt.xlabel("MJD")
plt.ylabel("Flux")
plt.title(f"Courbe lumière SNID {snid}")
plt.legend()
plt.show()

# -----------------------------
# 6️⃣ Construire un mini-dataset pour ML
# -----------------------------
# Feature simple : flux max par bande et MJD du pic
features_list = []
labels = []

for obj in head_data[:50]:  # petit sous-échantillon pour test
    snid = obj["SNID"]
    sn_type = obj["TYPE"]  # type de SN
    phot_obj = phot_data[phot_data["SNID"] == snid]
    feature_dict = {}
    for b in bands:
        mask = phot_obj["BAND"] == b
        if mask.sum() > 0:
            feature_dict[f"flux_max_{b}"] = np.max(phot_obj["FLUXCAL"][mask])
            feature_dict[f"mjd_peak_{b}"] = phot_obj["MJD"][mask][np.argmax(phot_obj["FLUXCAL"][mask])]
        else:
            feature_dict[f"flux_max_{b}"] = 0
            feature_dict[f"mjd_peak_{b}"] = 0
    features_list.append(feature_dict)
    labels.append(sn_type)

X = pd.DataFrame(features_list)
y = np.array(labels)
print("Features shape:", X.shape)
print("Labels:", np.unique(y))

# -----------------------------
# 7️⃣ Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -----------------------------
# 8️⃣ Entraîner un RandomForest simple
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# 9️⃣ Évaluer le modèle
# -----------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Ce que fait ce code :