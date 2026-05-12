"""
Modèles pour l'ajustement des courbes de lumière.
"""

import numpy as np


# --- Fonction de Bazin ---
def bazin_function(t, A, t0, t_fall, t_rise, B):
    """Fonction de Bazin pour ajuster une courbe de lumière."""
    # Éviter les divisions par zéro ou les valeurs extrêmes
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = (t - t0) / t_rise
        term1 = np.exp(-(t - t0) / np.maximum(t_fall, 1e-6))  # t_fall >= 1e-6
        term2 = 1 + np.exp(-x)
        return A * term1 / np.maximum(term2, 1e-6) + B  # Éviter division par zéro
