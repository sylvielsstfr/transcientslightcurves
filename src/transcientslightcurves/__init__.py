"""
Module pour l'analyse et la classification des courbes de lumière de supernovae.
"""
# Imports des sous-modules
# Version (gérée par setuptools_scm)
from .bazinfunction.classification import extract_features, predict_type, train_classifier
from .bazinfunction.fitting import fit_bazin_band, fit_single_event
from .bazinfunction.models import bazin_function
from .bazinfunction.utils import filter_valid_events, filter_valid_points

# Importer les constantes
from .constants import BANDS, ZERO_POINT
from .example_module import greetings, meaning
from .lib_elasticc2 import elasticc2_snana_reader

__all__ = ["greetings",
           "meaning",
           "elasticc2_snana_reader",
           "fit_bazin_band",
           "fit_single_event",
           "bazin_function",
           "filter_valid_events",
           "filter_valid_points",
           "extract_features",
           "predict_type",
           "train_classifier",
           "BANDS", "ZERO_POINT"
           ]
