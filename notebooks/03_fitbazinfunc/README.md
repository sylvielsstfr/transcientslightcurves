# Pipeline de classification de supernovae — fits Bazin (ELAsTiCC2)

Ce dossier contient le pipeline complet permettant de passer de courbes de
lumière simulées ELAsTiCC2 à un classifieur binaire **SNIa vs non-Ia**, en
utilisant des features extraites par ajustement de la fonction de Bazin sur
chaque bande photométrique.

## Vue d'ensemble du pipeline

```
03_elasticc2_fit_bazin_extract_features.ipynb
        │  (fit Bazin par bande, par objet)
        ▼
   features/bazin_features_<obj_class>.parquet   (un fichier par type de SN)
        │
        ├──────────────────────────────────────────────┐
        ▼                                                ▼
04_explore_features.ipynb                    05_build_train_test_sets.ipynb
  (exploration visuelle,                        (filtrage qualité, label,
   SNIa vs non-Ia, features brutes)               split stratifié z+label)
        │                                                │
        │                                                ▼
        │                                  train_test_sets/{train,test}_set.parquet
        │                                                │
        │                                                ▼
        │                                  06_train_classifier.ipynb
        │                                    (Random Forest + XGBoost,
        │                                     métriques, SHAP)
        │                                                │
        └──────────────── (lecture indépendante) ─────► models/
```

Les notebooks `01_` et `02_` (préexistants) explorent l'ajustement de Bazin sur
des courbes de lumière individuelles ; `03_` en est l'extension systématique à
un grand nombre d'objets, multi-classes, avec sauvegarde des features.

## Données d'entrée

Courbes de lumière simulées ELAsTiCC2, lues via `elasticc2_snana_reader`
(module `transcientslightcurves`), depuis :

```
/Users/dagoret/DATA/DESC_TD_PUBLIC/ELASTICC/ELASTICC2_TRAINING_SAMPLE_2/
├── ELASTICC2_TRAIN_02_SNIa-SALT3/
├── ELASTICC2_TRAIN_02_SNIb-Templates/
├── ELASTICC2_TRAIN_02_SNIc-Templates/
└── ELASTICC2_TRAIN_02_SNII-Templates/
```

---

## 1. `03_elasticc2_fit_bazin_extract_features.ipynb`

**Rôle** : pour chaque classe de supernova, sélectionner des objets, ajuster
la fonction de Bazin indépendamment dans chaque bande (`u, g, r, i, z, Y`), et
sauvegarder les features extraites dans un fichier parquet par classe.

### Modèle de Bazin

```
f(t) = A · exp[-(t-t0)/t_fall] / [1 + exp[-(t-t0)/t_rise]] + B
```

Référence : Dai, Kuhlmann, Wang & Kovacs (2017), *Photometric classification
and redshift estimation of LSST Supernovae*, [arXiv:1701.05689](https://arxiv.org/pdf/1701.05689).

### Paramètres principaux (valeurs actuelles dans le notebook)

| Paramètre | Valeur | Description |
|---|---|---|
| `OBJ_CLASS_LIST` | `SNIa-SALT3`, `SNIb-Templates`, `SNIc-Templates`, `SNII-Templates` | Classes SNANA traitées |
| `N_CURVES` | `100000` | Nombre max d'objets par classe (en pratique : tous les objets valides) |
| `Z_MIN` / `Z_MAX` | `0.05` / `2.5` | Intervalle de redshift (ZCMB) en présélection |
| `FILE_NUM` | `1` | Numéro du fichier PHOT chargé (1 à 40 ; `None` = tous) |
| `MIN_DETECTIONS` | `5` | Détections minimales par objet en présélection |
| `DETECTED_ONLY` | `True` | N'utilise que les points détectés pour le fit |
| `MIN_BANDS` / `MIN_POINTS` / `MIN_TOTAL_POINTS` | `3` / `3` / `5` | Critères de `filter_valid_events` |
| `RANDOM_SEED` | `42` | Graine de sélection aléatoire des objets |

⚠️ **Attention** : la classe SNII est nommée `SNII-Templates` dans
`OBJ_CLASS_LIST` (et dans les fichiers parquet produits), bien qu'une version
antérieure de ce notebook ait utilisé `SNII-Templateset`. Vérifier que ce nom
correspond bien au dossier réel sous `DATA_DIR` si le pipeline est ré-exécuté
sur un autre jeu de données.

### Flag qualité `is_good_fit`

Critères repris de Dai et al. (2017, §3.3), appliqués **par bande** puis
combinés en un flag global (vrai si toutes les bandes effectivement ajustées
par un Bazin complet — pas un fit constant — passent les coupures) :

- `t_rise > 1` (et pas trop proche de la borne, tolérance `0.01`)
- `-20 < B < 20`
- `χ²/ndof < 10`
- `t_fall < 150`
- `t_rise < t_fall`
- `A < 5000` (`A < 1000` pour les bandes `u` et `Y`)

Ce flag **n'est pas utilisé pour filtrer** dans ce notebook : il est
sauvegardé tel quel pour permettre un filtrage flexible en aval (notebook
`05_`).

### Sortie

Un fichier parquet par classe dans `features/` :

```
features/bazin_features_<obj_class>.parquet
```

Chaque ligne correspond à un objet et contient (~58 colonnes) :

- **Identifiants** : `SNID`, `obj_class`
- **Vérité** : `redshift` (ZCMB), `truth_GENTYPE`/`truth_SNTYPE`/`truth_PEAKMJD`/... (si
  disponibles dans le fichier de vérité SNANA)
- **Qualité globale** : `fit_success`, `chi2_total`, `ndof_total`, `chi2_red`,
  `is_good_fit`
- **Globaux** : `t_max_global`, `F_peak_global` (médiane des bandes réussies)
- **Par bande** (× 6 bandes) : `{band}_A`, `{band}_t0`, `{band}_t_fall`,
  `{band}_t_rise`, `{band}_B`, `{band}_t_max`, `{band}_f_max`, `{band}_m_p`,
  `{band}_chi2`, `{band}_ndof`, `{band}_chi2_red`, `{band}_success`
- **Couleurs au pic** : `c_ug`, `c_gr`, `c_ri`, `c_iz`, `c_zY` (différences de
  magnitude entre bandes adjacentes)

Note : beaucoup de bandes peuvent être `NaN` pour un objet donné (bande non
ajustée faute de points) — c'est attendu, en particulier pour `u` et `Y`,
moins bien échantillonnées par LSST.

### Résultat de la dernière exécution connue

| Classe | Objets chargés |
|---|---|
| SNIa-SALT3 | 1295 |
| SNIb-Templates | 204 |
| SNIc-Templates | 102 |
| SNII-Templates | 486 |
| **Total** | **2087** |

---

## 2. `04_explore_features.ipynb`

**Rôle** : exploration visuelle des features **brutes** (avant filtrage
qualité strict optionnel, et surtout **avant standardisation**), comparant
SNIa et non-Ia (Ib + Ic + II regroupés), pour repérer à l'œil les features les
plus discriminantes avant même d'entraîner un classifieur.

Se branche directement sur la sortie du notebook `03_` (tous les fichiers de
`features/`), indépendamment du split train/test.

### Contenu

1. Chargement + filtrage qualité (`REQUIRE_GOOD_FIT=True` par défaut) +
   construction du label binaire.
2. Regroupement des features en 4 catégories : Bazin par bande (48),
   couleurs (5), redshift (1), qualité du fit (4, indicatif).
3. Deux fonctions de tracé en grille (`plot_histograms_grid`,
   `plot_boxplots_grid`), robustes aux `NaN`.
4. Grilles histogrammes + boxplots pour chaque catégorie.
5. Tableau récapitulatif mean/std/median par label, et un **score de
   séparation** (type *d* de Cohen : écart de moyennes normalisé par
   l'écart-type combiné) classant les features les plus discriminantes.
6. Zoom (histogrammes + boxplots) sur le top 12 des features les plus
   séparatrices.

### Paramètres principaux

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `REQUIRE_GOOD_FIT` | `True` | Filtrer `is_good_fit==True` avant de tracer |
| `N_COLS` | `6` | Colonnes par grille de subplots |
| `BINS` | `25` | Nombre de bins des histogrammes |

Aucune sortie fichier — notebook purement exploratoire (figures inline).

---

## 3. `05_build_train_test_sets.ipynb`

**Rôle** : concaténer les fichiers de features par classe, filtrer les bons
fits, construire le label de classification, et produire un split
train/test **stratifié sur (label, bin de redshift)**.

### Pourquoi stratifier sur le redshift ?

Sans cela, train et test pourraient avoir des distributions de redshift
différentes pour une même classe, biaisant l'évaluation (un classifieur
testé sur un sous-échantillon plus proche/lointain que celui d'entraînement
donnerait des métriques trompeuses).

`safe_stratified_split` implémente un repli en cascade : essaie `MAX_Z_BINS`
bins de redshift (quantiles), réduit progressivement le nombre de bins si un
combo `(label, z_bin)` a moins de 2 membres, et en dernier recours stratifie
uniquement sur `label` (avec avertissement explicite).

### Paramètres principaux

| Paramètre | Valeur | Description |
|---|---|---|
| `TARGET_MODE` | `'binary'` | `'binary'` (SNIa vs non-Ia) ou `'multiclass'` |
| `REQUIRE_GOOD_FIT` | `True` | Ne garder que `is_good_fit == True` |
| `TEST_SIZE` | `0.2` | Fraction réservée au test |
| `MAX_Z_BINS` | `5` | Nombre de bins de redshift visé |
| `RANDOM_SEED` | `42` | Graine du split |

### Sortie

```
train_test_sets/train_set.parquet
train_test_sets/test_set.parquet
```

59 colonnes : 5 colonnes d'identification/label (`SNID`, `obj_class`,
`label`, `label_name`, `is_good_fit`, `z_bin`) + 54 colonnes de features
(48 paramètres Bazin + 5 couleurs + redshift).

### Résultat de la dernière exécution connue

- Filtrage qualité : **588 / 2087** objets conservés (`is_good_fit == True`)
  - SNIa-SALT3 : 386 · SNIb-Templates : 53 · SNIc-Templates : 24 · SNII-Templates : 125
- Split : **470 train / 118 test** (20.1 % test), stratification réussie
  avec **5 bins** de redshift
- Balance des labels quasi identique train/test (≈ 65.5 % / 34.5 % SNIa /
  non-Ia dans les deux échantillons)
- Redshift moyen cohérent entre train et test pour chaque label (ex. SNIa :
  ⟨z⟩≈0.42 dans les deux échantillons ; non-Ia : ⟨z⟩≈0.24–0.25)

---

## 4. `06_train_classifier.ipynb`

**Rôle** : entraîner et comparer un **Random Forest** et un **XGBoost** sur
les features standardisées, évaluer les performances, et analyser
l'importance des features (native Random Forest + SHAP).

### Imports protégés

`xgboost` et `shap` sont importés en `try/except ImportError` : le notebook
reste utilisable (Random Forest seul + importance native) même si l'une de
ces dépendances n'est pas installée. Message explicite (`pip install
xgboost` / `pip install shap`) en cas d'absence.

### Paramètres principaux

| Paramètre | Valeur | Description |
|---|---|---|
| `RANDOM_SEED` | `42` | Graine des modèles |
| `N_ESTIMATORS_RF` / `MAX_DEPTH_RF` | `200` / `10` | Hyperparamètres Random Forest |
| `N_ESTIMATORS_XGB` / `MAX_DEPTH_XGB` / `LEARNING_RATE_XGB` | `200` / `4` / `0.1` | Hyperparamètres XGBoost |

### Étapes

1. Chargement de `train_set.parquet` / `test_set.parquet`.
2. Préparation X/y : `fillna(0)` (les NaN représentent des bandes non
   ajustées) + `StandardScaler`.
3. Entraînement Random Forest (`class_weight='balanced'`) et XGBoost.
4. Comparaison des métriques : accuracy, precision, recall, F1, ROC-AUC.
5. Matrices de confusion côte à côte, courbes ROC superposées.
6. Importance des features :
   - Random Forest (diminution moyenne d'impureté), barplot top 20
   - SHAP (`TreeExplainer`, gère les formats de retour selon la version
     installée), barplot top 20 + beeswarm plot
7. Sauvegarde des modèles (`joblib`), du scaler, des métriques et des
   importances dans `models/` :

```
models/random_forest_model.joblib
models/feature_scaler.joblib
models/xgboost_model.joblib            (si xgboost disponible)
models/metrics_comparison.csv
models/rf_feature_importance.csv
models/shap_feature_importance.csv     (si shap disponible)
models/feature_columns.txt             (ordre exact des features à l'entraînement)
```

---

## Points d'attention / pistes d'amélioration

- **Beaucoup de NaN dans les features brutes** : la majorité des bandes ne
  sont pas ajustées pour la plupart des objets (visible dans l'aperçu de
  `train_set.parquet`). Cela peut réduire le pouvoir discriminant réel du
  classifieur — à surveiller via le notebook `04_` (groupes par bande) et la
  feature importance du notebook `06_`.
- **Déséquilibre des classes** : SNIa-SALT3 domine largement (386 vs 53+24+125
  = 202 non-Ia après filtrage qualité) — le `class_weight='balanced'` du
  Random Forest atténue ce biais, mais la performance sur les classes
  minoritaires (SNIc en particulier, 24 objets) reste à surveiller de près.
- **Mode multi-classe** : `TARGET_MODE='multiclass'` est implémenté dans
  `05_` mais le notebook `06_` est actuellement câblé pour le cas binaire
  (`CLASS_NAMES`, métriques `precision_score`/`recall_score` sans `average=`).
  L'adapter au multi-classe demanderait d'ajuster ces appels (`average='macro'`
  ou `'weighted'`, matrice de confusion N×N).
- **Cohérence du nom de classe SNII** : voir l'avertissement dans la section
  `03_` ci-dessus (`SNII-Templates` vs `SNII-Templateset`).
