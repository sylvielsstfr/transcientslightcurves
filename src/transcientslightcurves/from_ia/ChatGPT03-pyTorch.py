# # Classification avec pytorch

# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : 2025-11-06 (at NERSC **kernel desc-td-env-dev**)
# - last update : 2025-11-10 laptop on (**kernel pytorch-cpu-py312**)
# ## Prepare data


import os
import socket
import fitsio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




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


# ## Classify with pytorch

# - Super 😄 ! On va passer à une version PyTorch pour classifier les transients ELAsTiCC2 à partir des features extraites.
# - L’idée :
# - Dataset X et labels y (features comme flux_max, flux_mean, rise_time, etc.)
# - Encodage des labels en indices (SNIa-SALT3 → 0, SNIc → 1, …)
# - Mini réseau fully-connected (MLP)
# - Entraînement et évaluation avec PyTorch

# 🔹 Ce que ce notebook PyTorch fait :
# - Encode les labels en indices numériques (LabelEncoder).
# - Utilise MLP simple avec 2 couches cachées.
# - Optimiseur Adam, loss CrossEntropy.
# - Mini-batch pour un entraînement efficace même sur NERSC CPU ou GPU.
# - Affiche rapport de classification final avec précision par type de SN.

# ---------------------------
# 1️⃣ Préparer les données
# ---------------------------
X_values = X.values.astype(np.float32)
le = LabelEncoder()
y_values = le.fit_transform(y)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.3, random_state=42, stratify=y_values
)

# Convertir en tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ---------------------------
# 2️⃣ Définir le modèle
# ---------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = num_classes

model = MLPClassifier(input_dim, hidden_dim, output_dim)

# ---------------------------
# 3️⃣ Définir loss + optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4️⃣ Entraînement
# ---------------------------
epochs = 50
batch_size = 32

for epoch in range(epochs):
    permutation = torch.randperm(X_train_t.size()[0])
    epoch_loss = 0

    for i in range(0, X_train_t.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# ---------------------------
# 5️⃣ Évaluation
# ---------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    y_pred = torch.argmax(logits, dim=1).numpy()

print("Classification report :\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

