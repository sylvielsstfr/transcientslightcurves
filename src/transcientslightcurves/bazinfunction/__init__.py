# mon_module_sn/__init__.py
"""
Module pour l'analyse et la classification des courbes de lumière de supernovae.
"""


# --- Imports des sous-modules ---
from .classification import predict_type, train_classifier
from .fitting import fit_bazin_band, fit_single_event
from .models import bazin_function
from .utils import filter_valid_events

__all__ = ["predict_type",
           "train_classifier",
           "fit_bazin_band",
           "fit_single_event",
           "filter_valid_events","bazin_function"]

# --- Constantes globales ---
BANDS = ['u', 'g', 'r', 'i', 'z', 'Y']
ZERO_POINT = 25.0  # Point zéro pour LSST (système AB)

#
