# ## Classification Random Forest

# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : 2025-11-06 (at NERSC **kernel desc-td-env-dev**)
# - last update : 2025-11-10 laptop on (**kernel pytorch-cpu-py312**)

# ✅ Ce notebook fait tout :
#
# - Parcourt plusieurs types de SN et transients.
# - Lit HEAD + PHOT .FITS.gz compressés.
# - Filtre les bandes bidon '-'.
# - Extrait des features simples (flux max, MJD du pic, NOBS, redshift).
# - Visualise une courbe multibande d’exemple.

# In[1]:


# ===============================================
# Notebook Mini-ELAsTiCC2 : Lecture + Features + ML
# ===============================================

# 1️⃣ Import des librairies
import os,socket
import fitsio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix




def is_on_nersc():
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


# 2️⃣ Définir le chemin des données et types à analyser
if is_on_nersc():
    print("Configuration pour NERSC : utilisation des GPU ou des chemins spécifiques.")
    # Exemple : charger des données depuis un chemin NERSC
    BASE_PATH = "/global/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2"

else:
    print("Configuration pour laptop : utilisation des ressources locales.")
    # Exemple : charger des données depuis un chemin local
    BASE_PATH = "/Users/dagoret/DATA/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2"



sample_types = [
    "ELASTICC2_TRAIN_02_SNIa-SALT3",
    "ELASTICC2_TRAIN_02_SNIc-Templates",
    "ELASTICC2_TRAIN_02_SNIb-Templates"
]

# 3️⃣ Extraction de features
features_list = []
labels = []

for sn_type in sample_types:
    type_dir = os.path.join(BASE_PATH, sn_type)

    # Lister HEAD et PHOT
    head_files = sorted([f for f in os.listdir(type_dir) if "HEAD" in f])
    phot_files = sorted([f for f in os.listdir(type_dir) if "PHOT" in f])

    for head_file, phot_file in zip(head_files, phot_files):
        head_data = fitsio.FITS(os.path.join(type_dir, head_file))[1].read()
        phot_data = fitsio.FITS(os.path.join(type_dir, phot_file))[1].read()

        # Filtrer les bandes bidon et enlever espaces
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

        # Features globales depuis HEAD
        feature_dict["NOBS"] = head_data["NOBS"][0]
        feature_dict["REDSHIFT_FINAL"] = head_data["REDSHIFT_FINAL"][0]

        features_list.append(feature_dict)
        labels.append(sn_type)

# Convertir en DataFrame
X = pd.DataFrame(features_list)
y = np.array(labels)

print("Shape features :", X.shape)
print("Exemple features :\n", X.head())

# 4️⃣ Visualiser un objet (optionnel)
head_file = os.path.join(BASE_PATH, sample_types[0], sorted([f for f in os.listdir(os.path.join(BASE_PATH, sample_types[0])) if "HEAD" in f])[0])
phot_file = head_file.replace("HEAD", "PHOT")
phot_data = fitsio.FITS(phot_file)[1].read()

bands = set(b.strip() for b in phot_data["BAND"] if b.strip() != '-')
plt.figure(figsize=(8,5))
for b in bands:
    mask = np.array([x.strip() for x in phot_data["BAND"]]) == b
    plt.errorbar(phot_data["MJD"][mask], phot_data["FLUXCAL"][mask],
                 yerr=phot_data["FLUXCALERR"][mask], fmt='o', label=f'Band {b}')
plt.xlabel("MJD")
plt.ylabel("Flux")
plt.title("Courbe lumière multibande (exemple)")
plt.legend()
plt.show()

# 5️⃣ Split train/test et entraînement RandomForest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6️⃣ Prédiction et évaluation
y_pred = clf.predict(X_test)
print("Classification report :\n")
print(classification_report(y_test, y_pred))
print("Confusion matrix :\n")
print(confusion_matrix(y_test, y_pred))


# 🔹 Features enrichies + ML : Ce que cette version apporte

# - Plus de features par bande → plus de signal pour ML
# rise_time donne une idée de la montée de la lumière (utile pour différencier types de SN)
# - Toujours prêt à exécuter sur NERSC avec ELAsTiCC2 Training Sample 2
# - Compatible avec RandomForest, facilement échangeable avec d’autres modèles (XGBoost, PyTorch, …)


# ===============================================
# Notebook Mini-ELAsTiCC2 v2 : Features enrichies + ML
# ===============================================

import os
import fitsio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



features_list = []
labels = []

for sn_type in sample_types:
    type_dir = os.path.join(BASE_PATH, sn_type)
    head_files = sorted([f for f in os.listdir(type_dir) if "HEAD" in f])
    phot_files = sorted([f for f in os.listdir(type_dir) if "PHOT" in f])

    for head_file, phot_file in zip(head_files, phot_files):
        head_data = fitsio.FITS(os.path.join(type_dir, head_file))[1].read()
        phot_data = fitsio.FITS(os.path.join(type_dir, phot_file))[1].read()

        bands = set(b.strip() for b in phot_data["BAND"] if b.strip() != '-')
        feature_dict = {}

        for b in bands:
            mask = np.array([x.strip() for x in phot_data["BAND"]]) == b
            flux = phot_data["FLUXCAL"][mask]
            mjd = phot_data["MJD"][mask]

            if len(flux) > 0:
                flux_max = np.max(flux)
                flux_min = np.min(flux)
                flux_mean = np.mean(flux)
                flux_std = np.std(flux)

                # MJD du pic
                mjd_peak = mjd[np.argmax(flux)]

                # Rise time approximatif (tps entre premier et dernier pt >50% flux_max)
                half_max = flux_max / 2
                above_half = mjd[flux >= half_max]
                if len(above_half) >= 2:
                    rise_time = above_half[-1] - above_half[0]
                else:
                    rise_time = 0

                feature_dict.update({
                    f"flux_max_{b}": flux_max,
                    f"flux_min_{b}": flux_min,
                    f"flux_mean_{b}": flux_mean,
                    f"flux_std_{b}": flux_std,
                    f"mjd_peak_{b}": mjd_peak,
                    f"rise_time_{b}": rise_time
                })
            else:
                feature_dict.update({
                    f"flux_max_{b}": 0,
                    f"flux_min_{b}": 0,
                    f"flux_mean_{b}": 0,
                    f"flux_std_{b}": 0,
                    f"mjd_peak_{b}": 0,
                    f"rise_time_{b}": 0
                })

        # Features globales HEAD
        feature_dict["NOBS"] = head_data["NOBS"][0]
        feature_dict["REDSHIFT_FINAL"] = head_data["REDSHIFT_FINAL"][0]

        features_list.append(feature_dict)
        labels.append(sn_type)

X = pd.DataFrame(features_list)
y = np.array(labels)

print("Shape features :", X.shape)
print("Exemple features :\n", X.head())

# ---------------------------
# RandomForest Classifier
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification report :\n")
print(classification_report(y_test, y_pred))
print("Confusion matrix :\n")
print(confusion_matrix(y_test, y_pred))

