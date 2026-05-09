"""
Fonctions d'ajustement des courbes de lumière.
"""

from .bazin import fit_bazin_band, fit_single_event

__all__ = ["fit_bazin_band",
           "fit_single_event",
           ]
