#!/usr/bin/env python
# coding: utf-8

# # Classification avec pytorch RNN,LSTM,GRU

# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : 2025-11-07 (at NERSC **kernel desc-td-env-dev**)
# - update : 2025-11-09 : finish implementation at NERSC
# - last update : 2025-11-10 laptop on (**kernel pytorch-cpu-py312**)

# 1️⃣ Modèles séquentiels / temporels (RNN, LSTM, GRU)
# - Les courbes de lumière sont naturellement des séries temporelles. Au lieu de résumer tout en features, tu peux traiter la séquence complète par bande :
# - Input : MJD, flux, flux_err par bande
# - RNN / LSTM / GRU → apprend la dynamique des variations de flux dans le temps
# - Output : type de SN ou probabilité multi-classes
# - 💡 Avantages : capture les formes des courbes (rise, plateau, décroissance)
# - 💡 Inconvénients : plus gourmand en données et GPU, nécessite padding ou masking si les séquences ont des longueurs différentes

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from astropy.io import fits
import glob
import os
import socket
import random
import fitsio
import matplotlib.pyplot as plt
import sys


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


# ## Function definitions


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
    """ Load a lightcurve from HEAD and PHOT files into a DataFrame"""

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



# ============================================================
# Préparation des séquences (Δt, flux, band)
# ============================================================

band_to_idx = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'Y':5}

def make_sequence(df):
    """Build the sequence required by pytorch"""

    # remove dummy observations
    #df = df[df["BAND"] != "-"]
    df.drop(df[df["BAND"] == "-"].index, inplace=True)

    df = df.sort_values('MJD')
    mjd = df['MJD'].values
    flux = df['FLUXCAL'].values
    band = np.array([band_to_idx[b] for b in df['BAND'].values])

    # Δt : différence de temps normalisée
    dt = np.diff(mjd, prepend=mjd[0])
    dt = (dt - dt.mean()) / (dt.std() + 1e-6)

    # Normalisation du flux
    flux = (flux - flux.mean()) / (flux.std() + 1e-6)

    seq = torch.tensor(np.stack([dt, flux], axis=1), dtype=torch.float32)
    bands = torch.tensor(band, dtype=torch.long)
    return seq, bands


# ## Start here


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


# ### Load a number of SN samples

NF= 50
lightcurves = load_all_lightcurves(NF, dict_sntype_to_files)
print(f"{len(lightcurves)} curves loaded")


# ## Plot First Light Curve


index = 1
plot_lightcurve(lightcurves[index][0], title="SNID " + str(lightcurves[index][1]['SNID']))


# ### Generate sequences


sequences, bands_list, labels = [], [], []
for df, head in lightcurves:
    seq, bands = make_sequence(df)
    sequences.append(seq)
    bands_list.append(bands)
    labels.append(sample_types.index(df['LCTYPE'].unique()[0]))


# ### Generate paded sequences


lengths = torch.tensor([len(s) for s in sequences])
padded_seq = pad_sequence(sequences, batch_first=True)
padded_bands = pad_sequence(bands_list, batch_first=True, padding_value=-1)
labels = torch.tensor(labels)


# ### Dataset et DataLoader

# ============================================================
# Dataset et DataLoader
# ============================================================

class LightCurveDataset(Dataset):
    def __init__(self, seqs, bands, lengths, labels):
        self.seqs = seqs
        self.bands = bands
        self.lengths = lengths
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.seqs[i], self.bands[i], self.lengths[i], self.labels[i]



dataset = LightCurveDataset(padded_seq, padded_bands, lengths, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# ### Modèle avec embedding de bande et LSTM

# ============================================================
# Modèle avec embedding de bande et LSTM
# ============================================================

class LightCurveLSTM_Enhanced(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=1, num_classes=3, num_bands=6, band_emb_dim=4):
        super().__init__()
        self.band_emb = nn.Embedding(num_embeddings=num_bands, embedding_dim=band_emb_dim)
        self.lstm = nn.LSTM(input_dim + band_emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, bands, lengths):
        bands_clamped = bands.clamp(min=0)  # pour padding = -1
        emb = self.band_emb(bands_clamped)
        x = torch.cat([x, emb], dim=2)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.fc(h[-1])
        return out


# ### Entraînement
# ============================================================
# Entraînement
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightCurveLSTM_Enhanced(num_classes=len(sample_types)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, b, l, y in loader:
        x, b, l, y = x.to(device), b.to(device), l.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, b, l)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")


# ### Évaluation

# ============================================================
# Évaluation
# ============================================================
model.eval()
with torch.no_grad():
    out = model(padded_seq.to(device), padded_bands.to(device), lengths.to(device))
    pred = out.argmax(dim=1)
    acc = (pred.cpu() == labels).float().mean().item()

print(f"\n✅ Accuracy globale : {acc:.3f}")


# 💬 Explications clés

# | Élément                | Rôle                                                            |
# | ---------------------- | --------------------------------------------------------------- |
# | `Δt`                   | Encode l’irrégularité temporelle, essentiel pour le RNN         |
# | `flux`                 | Valeur photométrique normalisée                                 |
# | `band_emb`             | Transforme l’indice de bande (u,g,r,i,z,Y) en un vecteur appris |
# | `pack_padded_sequence` | Ignore les zéros de padding pour un apprentissage propre        |
# | `LSTM`                 | Capture la dynamique temporelle du flux                         |
# | `fc`                   | Prédit le type de SN à partir du dernier état caché             |
# ChatGPT a dit :
#
# Parfait 🌟
# Voici la version améliorée du modèle ELASTICC avec :
# un masque de padding explicite,
# une architecture modulaire qui peut basculer facilement entre LSTM, GRU, ou Transformer Encoder,
# une préparation des masques compatible avec les séquences irrégulières et padées.
# Tu pourras ainsi, sans changer tes données, tester des architectures modernes de traitement de séries temporelles astrophysiques.

# ============================================================
# Version améliorée du modèle : LSTM / GRU / Transformer
# avec masque de padding explicite
# ============================================================

class LightCurveModel(nn.Module):
    """
    Modèle pour séquences irrégulières de courbes de lumière :
      - encode les bandes via embedding
      - supporte LSTM, GRU ou TransformerEncoder
      - gère un masque de padding explicite
    """
    def __init__(self, 
                 input_dim=2,          # [Δt, flux]
                 band_emb_dim=4,       # embedding pour les bandes
                 hidden_dim=128,
                 num_layers=1,
                 num_classes=3,
                 num_bands=6,
                 model_type="lstm",    # "lstm", "gru" ou "transformer"
                 n_heads=4):
        super().__init__()
        self.model_type = model_type.lower()
        self.band_emb = nn.Embedding(num_embeddings=num_bands, embedding_dim=band_emb_dim)
        self.input_dim = input_dim + band_emb_dim

        if self.model_type == "lstm":
            self.rnn = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(self.input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=n_heads, batch_first=True)
            self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            hidden_dim = self.input_dim  # pour cohérence
        else:
            raise ValueError("model_type doit être 'lstm', 'gru' ou 'transformer'.")

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, bands, lengths):
        """
        x: (batch, seq_len, 2)
        bands: (batch, seq_len)
        lengths: (batch,)
        """
        # Embedding des bandes
        bands_clamped = bands.clamp(min=0)
        emb = self.band_emb(bands_clamped)
        x = torch.cat([x, emb], dim=2)

        # Masque de padding : True = doit être masqué
        max_len = x.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]

        if self.model_type in ["lstm", "gru"]:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.rnn(packed)
            if self.model_type == "lstm":
                _, (h, _) = hidden
            else:
                _, h = None, hidden
            out = self.fc(h[-1])
        else:  # Transformer
            # TransformerEncoder attend mask=(batch, seq_len)
            out_seq = self.rnn(x, src_key_padding_mask=mask)
            # On prend le dernier état non masqué pour chaque séquence
            last_indices = (lengths - 1).clamp(min=0)
            last_outputs = out_seq[torch.arange(len(last_indices)), last_indices]
            out = self.fc(last_outputs)

        return out


# ###  ⚙️ Utilisation (LSTM, GRU, Transformer)
# Exemple d’instanciation :


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choisir le type de modèle :
model = LightCurveModel(
    input_dim=2, hidden_dim=128, num_layers=2,
    num_classes=len(sample_types), num_bands=6, 
    model_type="transformer"   # ou "lstm" / "gru"
).to(device)


# ### Entraînement (inchangé)
# Tu peux réutiliser exactement la même boucle d’entraînement que précédemment :

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, b, l, y in loader:
        x, b, l, y = x.to(device), b.to(device), l.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, b, l)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")


# ### Evaluation

model.eval()
with torch.no_grad():
    out = model(padded_seq.to(device), padded_bands.to(device), lengths.to(device))
    pred = out.argmax(dim=1)
    acc = (pred.cpu() == labels).float().mean().item()

print(f"✅ Accuracy globale : {acc:.3f}")


# ### 💡 Avantages de cette version

# | Fonctionnalité                         | Description                                                 |
# | -------------------------------------- | ----------------------------------------------------------- |
# | ✅ **Masque explicite**                 | les pas de temps “padés” sont ignorés par le modèle         |
# | 🔄 **Modèle interchangeable**          | `model_type="lstm"`, `"gru"` ou `"transformer"`             |
# | 🧠 **Embeddings de bandes**            | apprend à associer des patterns photométriques par filtre   |
# | 🔥 **Transformer compatible**          | gère naturellement les longueurs variables via le masque    |
# | 🪶 **Sortie adaptée à classification** | le dernier état (non masqué) de chaque séquence est utilisé |
#
# Tu as maintenant un pipeline complet pour travailler sur les courbes de lumière irrégulières avec PyTorch, et tu peux expérimenter LSTM, GRU ou Transformer avec embeddings de bandes et masques de padding.
# Prends le temps d’explorer chaque étape : préparation des séquences, normalisation, embeddings, et architectures. Ça te donnera une base solide pour classifier ou prédire des flux temporels.
#
# Si tu veux, la prochaine fois on pourra aller encore plus loin avec positional encoding basé sur Δt pour Transformers, ou même des Neural ODEs pour séries temporelles continues. 🚀
#
# Bon courage pour l’étude, et tu auras un super modèle astrophysique prêt à l’emploi ! 🌌
#
# Veux‑tu que je te prépare un petit résumé visuel de ce pipeline pour t’aider à t’y retrouver plus facilement quand tu reviendras ?

# - Parfait 🌟, voici un résumé visuel et conceptuel du pipeline RNN/LSTM/Transformer pour les courbes de lumière irrégulières.

# 
#
# ┌──────────────────────────┐
# │ Chargement des fichiers  │
# │  - HEAD.FITS.gz          │
# │  - PHOT.FITS.gz          │
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Extraction DataFrame     │
# │  Colonnes :              │
# │  - MJD                   │
# │  - FLUXCAL / FLUXCALERR  │
# │  - BAND                  │
# │  - SNID, LCTYPE          │
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Préparation séquences     │
# │  Pour chaque courbe :     │
# │  - Trier par MJD          │
# │  - Calcul Δt = diff(MJD)  │
# │  - Normaliser flux        │
# │  - Convertir band → idx   │
# │  - Tensor : [Δt, flux]   │
# │  - Tensor bands           │
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Padding pour batch        │
# │  - pad_sequence           │
# │  - lengths                │
# │  - mask_padding (True=pad)│
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Modèle PyTorch            │
# │  - Embedding bandes       │
# │  - LSTM / GRU / Transformer│
# │  - Input = [Δt, flux, band_emb]│
# │  - Gestion padding/masque │
# │  - Dernier hidden → FC    │
# │  - Output = probabilité classe SN │
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Entraînement              │
# │  - CrossEntropyLoss       │
# │  - Optimizer : Adam       │
# │  - Batch DataLoader       │
# │  - Epochs = 10–20         │
# └─────────┬────────────────┘
#           │
#           ▼
# ┌──────────────────────────┐
# │ Évaluation               │
# │  - Prédiction sur séquences|
# │  - Accuracy / métriques   │
# └──────────────────────────┘
#
#
# ### Notes clés
#
# - Δt : encode la dimension temporelle irrégulière.
#
# - Embeddings de bandes : permettent au modèle de distinguer les patterns par filtre photométrique.
#
# - Masque de padding : ignore les valeurs artificielles pour LSTM/GRU ou Transformer.
#
# - Modularité du modèle : tu peux changer LSTM ↔ GRU ↔ Transformer sans changer la préparation des données.
#
# - Flux normalisé : essentiel pour stabiliser l’apprentissage.
