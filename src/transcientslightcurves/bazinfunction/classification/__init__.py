"""
Sous-module pour la classification des supernovae.
"""
from .classifier import extract_features, predict_type, train_classifier

__all__ = ["extract_features", "predict_type", "train_classifier"]
