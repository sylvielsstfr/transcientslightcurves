#!/usr/bin/env python
# coding: utf-8

# # Classification avec pytorch RNN,LSTM,GRU

# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : 2025-11-06 (at NERSC **kernel desc-td-env-dev**)
# - update : 2025-11-08 at NERSC
# - last update : 2025-11-10 laptop on (**kernel pytorch-cpu-py312**)

# 1️⃣ Modèles séquentiels / temporels (RNN, LSTM, GRU)
#
# - Les courbes de lumière sont naturellement des séries temporelles. Au lieu de résumer tout en features, tu peux traiter la séquence complète par bande :
# - Input : MJD, flux, flux_err par bande
# - RNN / LSTM / GRU → apprend la dynamique des variations de flux dans le temps
# - Output : type de SN ou probabilité multi-classes
# - 💡 Avantages : capture les formes des courbes (rise, plateau, décroissance)
# - 💡 Inconvénients : plus gourmand en données et GPU, nécessite padding ou masking si les séquences ont des longueurs différentes


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from astropy.io import fits
import random
import fitsio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
import os
import socket
import glob


class SN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: hidden state final
        out = self.fc(hn[-1])
        return out


# - input_dim = 2 ou 3 (flux, flux_err, optionnellement bande encodée)
# - hidden_dim : 32–128 suffisent pour commencer
# - Prétraitement : séquences padding ou découpées à longueur fixe

# 2️⃣ Transformers / LLM pour séries temporelles
# - Les transformers (ou petits LLMs) commencent à être utilisés pour les lightcurves :
# - Input : séquence de tokens ou vecteurs (flux, MJD, bande)
# - Attention apprend à se concentrer sur les points clés (pic, montée, décroissance)
# - Output : classification multi-classes
# - Avantages : meilleure capture des relations longues distances dans la série.
# - Inconvénients : lourd, nécessite GPU, tuning complexe.
# - Des implémentations existent sur Hugging Face :
# - Modèles Time Series Transformer
# - AstroNet et SuperNNova pour transients

# 3️⃣ Approche hybride
# - Tu peux combiner features classiques + séquence complète :
# - Extraire features globales (flux_max, flux_mean, rise_time)
# - Coupler à une LSTM ou transformer pour capturer la dynamique
# - Fusionner dans un réseau final (MLP) pour la classification

# 💡 Suggestion pour ton notebook NERSC :
# - Commencer avec MLP sur features (déjà fait)
# - Ensuite, préparer les séquences : MJD/flux par bande, padding, mask
# - Tester LSTM/GRU sur un sous-échantillon pour vérifier performance et temps GPU
# - Enfin, explorer Transformers si tu veux vraiment capturer les patterns complexes

# ## Prepare data


print(sys.executable)


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



def buildDictObjTypeFromFile(path_top,objtypelist):
    """
    Retrun a dictionnary with the type of object as key and values are dictionary of ELASTIC2 list of header file and phtometric files
    """

    dict_objtype_to_files = {}

    # loop on object type
    for obj_type in objtypelist:
        path = os.path.join(path_top, obj_type)

        head_files = sorted(glob.glob(os.path.join(path, "*_HEAD.FITS.gz")))
        phot_files = sorted(glob.glob(os.path.join(path, "*_PHOT.FITS.gz")))
        dict_objtype_to_files[obj_type] = {"head_files" : head_files, "phot_files": phot_files }
    return dict_objtype_to_files


def load_lightcurve(head_file, phot_file, lctype):
    import pandas as pd
    from astropy.io import fits

    head = fits.open(head_file)[1].data
    phot = fits.open(phot_file)[1].data

    # Récupère l’identifiant SNID depuis le HEAD uniquement
    snid = str(head["SNID"][0]).strip()

    # Création du DataFrame photométrique
    df = pd.DataFrame({
        "SNID": snid,
        "LCTYPE": lctype,
        "MJD": phot["MJD"],
        "BAND": [b.strip() for b in phot["BAND"]],
        "FLUXCAL": phot["FLUXCAL"],
        "FLUXCALERR": phot["FLUXCALERR"]
    })

    # ignorer les bandes bidon
    #df.drop(df[df["BAND"] == "-"].index, inplace=True)

    # Forcer le little-endian pour toutes les colonnes float ou int
    for col in ["MJD", "FLUXCAL", "FLUXCALERR"]:
        df[col] = df[col].astype(df[col].dtype.newbyteorder('<'))

    return df, head


def load_all_lightcurves(nf, dict_objtype_tofile):
    """
    """

    lightcurves = []
    for obj_type, filesdict in dict_objtype_tofile.items() :
        head_files = filesdict["head_files"]
        phot_files = filesdict["phot_files"]
        for h, p in zip(head_files[:nf], phot_files[:nf]):
            try:
                df, head = load_lightcurve(h, p, obj_type)
                lightcurves.append((df, head))
            except Exception as e:
                print(f"Erreur avec {h}: {e}")
    return lightcurves



def plot_lightcurve(df, title=None):
    plt.figure(figsize=(10, 5))
    bands = sorted(df['BAND'].unique())
    for band in bands:
        if band.strip() == '-':  # ignorer les bandes bidon
            continue
        d = df[df['BAND'] == band]
        plt.errorbar(d['MJD'], d['FLUXCAL'], yerr=d['FLUXCALERR'], fmt='o', label=band.strip())
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


# ## Start


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


# Répertoire d'un type de SN
sn_type = "ELASTICC2_TRAIN_02_SNIa-SALT3"
sample_types = [
    "ELASTICC2_TRAIN_02_SNIa-SALT3",
    "ELASTICC2_TRAIN_02_SNIc-Templates",
    "ELASTICC2_TRAIN_02_SNIb-Templates"]


dict_sntype_to_files = buildDictObjTypeFromFile(path_top = BASE_PATH,objtypelist = sample_types )



# -----------------------------
# 3️⃣ Lire un HEAD aléatoire
# -----------------------------
# select the first type of SN
sntype = list(dict_sntype_to_files.keys())[0]
PATH = os.path.join(BASE_PATH,sntype)

#then specify the path
head_files = dict_sntype_to_files[sntype]["head_files"]

head_file = os.path.join(PATH, random.choice(head_files))
head_data = fitsio.FITS(head_file)[1].read()
print("Colonnes HEAD :", head_data.dtype.names)
print("Exemple HEAD :\n", head_data[:1])



NF= 50
lightcurves = load_all_lightcurves(NF, dict_sntype_to_files)
print(f"{len(lightcurves)} curves loaded")


print(lightcurves[0])


index = 1
plot_lightcurve(lightcurves[index][0], title="SNID " + str(lightcurves[index][1]['SNID']))


# ## Add extracted features

def extract_features(df):
    bands = ['u', 'g', 'r', 'i', 'z', 'Y']
    features = []
    for band in bands:
        d = df[df['BAND'].str.strip() == band]
        if len(d) == 0:
            features.extend([np.nan, np.nan, np.nan, np.nan])
        else:
            flux = d['FLUXCAL'].values
            mjd = d['MJD'].values
            features.extend([
                np.max(flux),
                mjd[np.argmax(flux)],
                np.mean(flux),
                np.std(flux)
            ])
    return np.array(features)



lc = lightcurves[0]



X = np.array([extract_features(lc[0]) for lc in lightcurves])
y = np.array([lc[0]['LCTYPE'].unique()[0] for lc in lightcurves])  # ou le label que tu as


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


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
X_values = X.astype(np.float32)
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


# ## Etude préalable pour utiliser pytorch

# ### 🧠 1. Concept général
# Une courbe de lumière = une séquence de luminosités (ou magnitudes) observées à des temps donnés.
# Le but du RNN est de modéliser la dynamique temporelle :
# soit pour prédire les valeurs futures,
# soit pour classifier le type d’objet (e.g. SN Ia vs SN II),
# soit pour extraire un encodage latent utile à d’autres tâches.

# ### 🧩 2. Préparer les données
# Les RNNs nécessitent des séquences normalisées et de même longueur (ou padées).



# Supposons que chaque courbe soit une liste de [temps, flux]
curves = [
    torch.tensor([[0., 1.0], [1., 1.2], [2., 0.9]]),    # 3 points
    torch.tensor([[0., 0.5], [1., 0.7]])               # 2 points
]

# On pad les séquences à la même longueur
padded_curves = pad_sequence(curves, batch_first=True)  # shape: (batch, seq_len, features)
lengths = torch.tensor([len(c) for c in curves])

print(padded_curves)
print(lengths)


# 👉 Si les pas de temps sont irréguliers, tu peux soit :
# - interpoler les flux sur une grille de temps régulière ;
# - soit fournir Δt comme feature d’entrée (pratique pour RNNs physiques).

# ⚙️ 3. Définir le modèle RNN
# - Voici une version de base avec nn.RNN, nn.LSTM ou nn.GRU.
# - Pour la plupart des applications astrophysiques, LSTM ou GRU marchent mieux que RNN simple.
# - Exemple : LSTM pour classification


class LightCurveRNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # pack la séquence pour ignorer les paddings
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        # h contient le dernier état caché de chaque couche
        out = self.fc(h[-1])  # dernier hidden state de la dernière couche
        return out


# 🔁 4. Entraînement
# Exemple pour une tâche de classification :


model = LightCurveRNN(input_dim=2, hidden_dim=64, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(padded_curves, lengths)
    labels = torch.tensor([0, 1])  # exemple de classes
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")


# 🧭 5. Variantes utiles pour courbes de lumière

# | Cas d’usage                          | Adaptation possible                            |
# | ------------------------------------ | ---------------------------------------------- |
# | Données multi-bandes (g, r, i, z, y) | input_dim = 1 + nb_bandes                      |
# | Données irrégulières                 | ajouter Δt en feature                          |
# | Bruit variable                       | ajouter σ_flux en feature                      |
# | Séries longues                       | utiliser GRU ou LSTM bidirectionnel            |
# | Encodage pour IA bayésienne          | utiliser le dernier `h` comme embedding latent |


# 🌌 6. Exemple plus réaliste : séquences multi-bandes


# Features = [Δt, flux_g, flux_r, flux_i, flux_z, flux_y]
input_dim = 6
model = LightCurveRNN(input_dim=input_dim, hidden_dim=128, num_classes=5)


# 🧩 7. (Optionnel) Prédiction de flux futurs
# Tu peux modifier la tête du réseau pour sortir une valeur temporelle suivante au lieu d’une classe :

# self.fc = nn.Linear(hidden_dim, output_dim)  # output_dim = nb_bandes par ex.


# 🚀 Bonus : alternatives modernes
# Pour des performances supérieures :
# - Transformer temporel (nn.TransformerEncoder) avec masques pour séquences irrégulières ;
# - Neural ODEs ou Temporal Convolutional Networks si tu veux modéliser le temps continu.

# Tu lis les courbes de lumière multi-bandes (u,g,r,i,z,Y) depuis ELASTICC2, où chaque fichier contient des observations à des temps irréguliers (MJD).
# Mais ton extract_features() résume chaque bande par des statistiques globales — ce qui perd toute la structure temporelle.
# 👉 Si ton objectif est de tirer parti de la dynamique temporelle complète, il faut au contraire garder les points individuels dans une séquence, et adapter le RNN à l’irrégularité temporelle.

# ### 🎯 Objectif
# Construire un LSTM PyTorch qui prend en entrée des séquences de points [Δt, flux, band]
# pour classifier la supernova (LCTYPE).

# #### 🧩 Étape 1 — Préparer les séquences temporelles
# Chaque courbe doit être une séquence temporelle ordonnée par MJD, avec le pas de temps Δt = MJD[i] - MJD[i-1].


band_to_idx = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'Y':5}

def make_sequence(df):

    # remove dummy observations
    #df = df[df["BAND"] != "-"]
    df.drop(df[df["BAND"] == "-"].index, inplace=True)


    # Trie par temps
    df = df.sort_values('MJD')
    mjd = df['MJD'].values
    flux = df['FLUXCAL'].values
    band = np.array([band_to_idx[b.strip()] for b in df['BAND'].values])

    # Δt : différence de temps normalisée (premier = 0)
    dt = np.diff(mjd, prepend=mjd[0])
    dt = (dt - dt.mean()) / (dt.std() + 1e-6)

    # Flux normalisé par écart-type
    flux = (flux - flux.mean()) / (flux.std() + 1e-6)

    # Une feature par point : [Δt, flux, band_index]
    seq = np.stack([dt, flux, band], axis=1)
    return torch.tensor(seq, dtype=torch.float32)


# Ensuite, on crée un jeu de séquences pour toutes les courbes :

sequences = [make_sequence(lc[0]) for lc in lightcurves]
labels = [sample_types.index(lc[0]['LCTYPE'].unique()[0]) for lc in lightcurves]

# Padding pour obtenir un batch
lengths = torch.tensor([len(s) for s in sequences])
padded_sequences = pad_sequence(sequences, batch_first=True)
labels = torch.tensor(labels)


print("labels : ",labels)
print("padded_sequences : ",padded_sequences)
print("lengths : ",lengths)


# #### ⚙️ Étape 2 — Définir le modèle PyTorch
# Un LSTM simple qui ingère [Δt, flux, band] et sort une probabilité par type :


class LightCurveLSTM(nn.Module):
    #def __init__(self, input_dim=3, hidden_dim=64, num_layers=1, num_classes=3):
    def __init__(self, input_dim=3, hidden_dim=4, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.fc(h[-1])
        return out


# #### 🔁 Étape 3 — Entraînement


dataset = TensorDataset(padded_sequences, lengths, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = LightCurveLSTM(input_dim=3, hidden_dim=64, num_classes=len(sample_types))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


## Too slow
if 1:
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, ll, y in loader:
            print(x[0],ll[0],y[0])
            optimizer.zero_grad()
            out = model(x, ll)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")


# #### 🧭 Étape 4 — Prédiction / évaluation


if 1:
    model.eval()
    with torch.no_grad():
        out = model(padded_sequences, lengths)
        pred = out.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
        print(f"Accuracy: {acc:.2f}")


# 💡 Points importants
# - Δt encode l’irrégularité temporelle → le réseau apprend à en tenir compte.
# - band est codé en entier, mais tu peux le transformer en embedding pour améliorer la performance :
# self.band_emb = nn.Embedding(num_embeddings=6, embedding_dim=3)
# - puis concaténer l’embedding à [Δt, flux] avant d’entrer dans le LSTM.
# - Tu peux ensuite utiliser des architectures plus puissantes :
# - GRU (plus stable que LSTM sur petits jeux),
# - Transformer temporel avec masque sur Δt,
# - ou des Neural ODEs pour le temps continu.

# ### 😎 — allons-y pour une version robuste et pratique du modèle PyTorch, spécialement adaptée à tes courbes ELASTICC irrégulières et multi-bandes.
# Voici un pipeline complet : préparation des séquences + modèle avec embedding des bandes + entraînement robuste avec padding et masques.

# #### 🧩 Étape 1 — Préparation des séquences avec embedding des bandes
# Chaque point de la courbe est décrit par :
# - Δt → le pas de temps (irrégulier),
# - flux → le flux calibré (normalisé),
# - band → la bande photométrique (u,g,r,i,z,Y).


band_to_idx = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'Y':5}

def make_sequence_v2(df):
    """Build the sequence required by pytorch

    :param df: _description_
    :type df: _type_
    :return: _description_
    :rtype: _type_
    """

    # remove dummy observations
    #df = df[df["BAND"] != "-"]
    df.drop(df[df["BAND"] == "-"].index, inplace=True)


    df = df.sort_values('MJD')
    mjd = df['MJD'].values
    flux = df['FLUXCAL'].values
    band = np.array([band_to_idx[b.strip()] for b in df['BAND'].values])

    # Δt = temps entre observations successives (normalisé)
    dt = np.diff(mjd, prepend=mjd[0])
    dt = (dt - np.mean(dt)) / (np.std(dt) + 1e-6)

    # Flux normalisé par écart-type
    flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-6)

    # Conversion en tenseurs torch
    seq = torch.tensor(np.stack([dt, flux], axis=1), dtype=torch.float32)
    bands = torch.tensor(band, dtype=torch.long)
    return seq, bands


# Ensuite :


sequences, bands_list, labels = [], [], []
for df, head in lightcurves:
    seq, bands = make_sequence_v2(df)
    sequences.append(seq)
    bands_list.append(bands)
    labels.append(sample_types.index(df['LCTYPE'].unique()[0]))

lengths = torch.tensor([len(s) for s in sequences])
padded_seq = pad_sequence(sequences, batch_first=True)   # (batch, seq_len, 2)
padded_bands = pad_sequence(bands_list, batch_first=True, padding_value=-1)  # (batch, seq_len)
labels = torch.tensor(labels)


# #### ⚙️ Étape 2 — Modèle avec embedding de bande et masque de padding


class LightCurveLSTM_Enhanced(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=1, num_classes=3, num_bands=6, band_emb_dim=4):
        super().__init__()
        # Embedding pour les bandes photométriques
        self.band_emb = nn.Embedding(num_embeddings=num_bands, embedding_dim=band_emb_dim, padding_idx=-1)
        self.lstm = nn.LSTM(input_dim + band_emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, bands, lengths):
        # bands: (batch, seq_len)
        emb = self.band_emb(bands.clamp(min=0))  # clamp car padding_idx = -1
        x = torch.cat([x, emb], dim=2)  # concatène les features

        # pack la séquence pour ignorer le padding
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.fc(h[-1])
        return out


# #### 🔁 Étape 3 — Entraînement robuste avec DataLoader


class LightCurveDataset(Dataset):
    def __init__(self, seqs, bands, lengths, labels):
        self.seqs = seqs
        self.bands = bands
        self.lengths = lengths
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.seqs[i], self.bands[i], self.lengths[i], self.labels[i]

dataset = LightCurveDataset(padded_seq, padded_bands, lengths, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# #### 🚀 Étape 4 — Boucle d’entraînement


model = LightCurveLSTM_Enhanced(
    input_dim=2, hidden_dim=128, num_classes=len(sample_types),
    num_bands=6, band_emb_dim=4
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    total_loss = 0
    for x, b, l, y in loader:
        optimizer.zero_grad()
        out = model(x, b, l)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")


# #### 📊 Étape 5 — Évaluation simple


model.eval()
with torch.no_grad():
    out = model(padded_seq, padded_bands, lengths)
    pred = out.argmax(dim=1)
    acc = (pred == labels).float().mean().item()
print(f"Accuracy globale : {acc:.3f}")


# #### ✨ Améliorations possibles

# | Amélioration                          | Commentaire                                                          |
# | ------------------------------------- | -------------------------------------------------------------------- |
# | 🔁 **GRU**                            | remplace `nn.LSTM` par `nn.GRU`, souvent plus stable sur petits jeux |
# | ⏱️ **masque temporel**                | utile si certaines bandes manquent longtemps                         |
# | 🧠 **Transformers temporels**         | type “Time Transformer” (pour données irrégulières)                  |
# | 🧩 **ajouter flux_err comme feature** | améliore la robustesse au bruit                                      |
# | 🪄 **Data augmentation**              | bruit aléatoire sur flux, décale les temps, etc.                     |

