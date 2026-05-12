"""
Filtres pour les données de courbes de lumière.
"""


def filter_valid_points(ltcv_df):
    """
    Filtre les points valides d'une courbe de lumière :
    - FLUXCALERR > 0
    - FLUXCAL > 0
    - SNR > 3
    """
    mask_det = (
        (ltcv_df['FLUXCALERR'] > 0) &
        (ltcv_df['FLUXCAL'] > 0) &
        (ltcv_df['FLUXCAL'] / ltcv_df['FLUXCALERR'] > 3)
    )
    return ltcv_df[mask_det].copy()

def filter_valid_events(subset, all_ltcvs, min_bands=2, min_points=2, min_total_points=5):
    """
    Filtre les événements avec :
    - Au moins `min_bands` bandes avec ≥ `min_points` points.
    - Au moins `min_total_points` points au total.
    """
    valid_snids = []
    for snid in subset['SNID'].unique():
        ltcv = all_ltcvs[all_ltcvs['SNID'] == snid]
        ltcv = filter_valid_points(ltcv)

        if len(ltcv) < min_total_points:
            continue

        band_counts = ltcv['BAND'].value_counts()
        valid_bands = band_counts[band_counts >= min_points]
        if len(valid_bands) >= min_bands:
            valid_snids.append(snid)

    return subset[subset['SNID'].isin(valid_snids)]

