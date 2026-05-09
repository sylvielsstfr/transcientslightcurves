"""
Ajustement des courbes de lumière avec la fonction de Bazin.
"""

import numpy as np
from scipy.optimize import curve_fit

from ...constants import BANDS, ZERO_POINT  # ✅ Importe depuis constants.py
from ..models import bazin_function


def fit_bazin_band(t, flux, flux_err, zero_point=ZERO_POINT):
    """
    Ajuste la fonction de Bazin sur une courbe de lumière d'une seule bande.
    """
    if len(t) < 2:
        return {
            'success': False,
            'error': 'Pas assez de points (min 2 requis).',
            'params': None,
            'chi2': np.nan,
            'ndof': np.nan,
            'pcov': None
        }

    # Filtrer les flux négatifs ou NaN
    mask = (flux > 0) & (flux_err > 0) & ~np.isnan(flux) & ~np.isnan(flux_err)
    if np.sum(mask) < 2:
        return {
            'success': False,
            'error': 'Trop de points invalides (flux <= 0 ou NaN).',
            'params': None,
            'chi2': np.nan,
            'ndof': np.nan,
            'pcov': None
        }
    t, flux, flux_err = t[mask], flux[mask], flux_err[mask]

    p0 = [np.max(flux), t[np.argmax(flux)], 15.0, 5.0, 0.0]
    bounds = ([0, -np.inf, 1e-6, 1e-6, -20], [np.inf, np.inf, 100, 50, 20])  # t_fall >= 1e-6, t_rise >= 1e-6

    try:
        popt, pcov = curve_fit(
            bazin_function, t, flux,
            p0=p0, bounds=bounds, sigma=flux_err, absolute_sigma=True
        )
        A, t0, t_fall, t_rise, B = popt

        # Vérifier que les paramètres sont physiques
        if A <= 0 or t_fall <= 1e-6 or t_rise <= 1e-6:
            return {
                'success': False,
                'error': f'Paramètres non physiques: A={A:.2f}, t_fall={t_fall:.2f}, t_rise={t_rise:.2f}.',
                'params': None,
                'chi2': np.nan,
                'ndof': np.nan,
                'pcov': pcov
            }

        model_flux = bazin_function(t, *popt)
        chi2 = np.sum(((flux - model_flux) / flux_err) ** 2)
        ndof = len(t) - 5  # 5 paramètres : A, t0, t_fall, t_rise, B

        if ndof <= 0:
            return {
                'success': False,
                'error': f'ndof <= 0 (len(t)={len(t)}).',
                'params': None,
                'chi2': chi2,
                'ndof': ndof,
                'pcov': pcov
            }

        # Calcul de t_max et f_max
        if t_fall / t_rise > 1:
            t_max = t0 + t_rise * np.log(t_fall / t_rise - 1)
            x = t_rise / t_fall
            if x <= 0 or x >= 1:
                return {
                    'success': False,
                    'error': f'x = t_rise / t_fall = {x:.2f} non valide (doit être 0 < x < 1).',
                    'params': None,
                    'chi2': chi2,
                    'ndof': ndof,
                    'pcov': pcov
                }
            f_max = A * (x ** x) * ((1 - x) ** (1 - x)) + B
            m_p = -2.5 * np.log10(f_max) + zero_point
        else:
            t_max, f_max, m_p = np.nan, np.nan, np.nan

        return {
            'success': True,
            'params': {
                'A': A, 't0': t0, 't_fall': t_fall,
                't_rise': t_rise, 'B': B, 't_max': t_max,
                'f_max': f_max, 'm_p': m_p
            },
            'chi2': chi2,
            'ndof': ndof,
            'pcov': pcov,
            'error': None
        }

    except RuntimeError as e:
        return {
            'success': False,
            'error': f"Échec de l'ajustement : {str(e)}",
            'params': None,
            'chi2': np.nan,
            'ndof': np.nan,
            'pcov': None
        }

def fit_single_event(ltcv_df, ref_band='r'):
    """
    Ajuste un événement (supernova) avec la fonction de Bazin pour chaque bande.
    """
    from ..utils.filters import filter_valid_points  # Importe depuis utils/

    # Filtrer les points valides
    df = filter_valid_points(ltcv_df)

    if len(df) < 2:
        return {'success': False, 'error': 'Pas assez de points valides.'}

    band_results = {}
    chi2_total = 0.0
    ndof_total = 0
    successful_bands = []

    for band in BANDS:
        band_mask = df['BAND'] == band
        if band_mask.sum() < 2:
            if band_mask.sum() == 1:
                B = df.loc[band_mask, 'FLUXCAL'].values[0]
                band_results[band] = {
                    'success': True,
                    'params': {'A': 0, 't0': 0, 't_fall': 0, 't_rise': 0, 'B': B, 't_max': np.nan, 'f_max': B, 'm_p': np.nan},
                    'chi2': 0,
                    'ndof': 0,
                    'pcov': None,
                    'error': "Ajustement constant (1 point)."
                }
                successful_bands.append(band)
            else:
                band_results[band] = {'success': False, 'error': f'Pas assez de points pour la bande {band}.', 'params': None}
            continue

        t = df.loc[band_mask, 'MJD'].values
        flux = df.loc[band_mask, 'FLUXCAL'].values
        flux_err = df.loc[band_mask, 'FLUXCALERR'].values

        result = fit_bazin_band(t, flux, flux_err)
        band_results[band] = result

        if result['success']:
            chi2_total += result['chi2']
            ndof_total += result['ndof']
            successful_bands.append(band)

    if not successful_bands:
        return {'success': False, 'error': 'Aucune bande n\'a pu être ajustée.'}

    t_max_values = [band_results[band]['params']['t_max'] for band in successful_bands if not np.isnan(band_results[band]['params']['t_max'])]
    if not t_max_values:
        return {'success': False, 'error': 'Aucun t_max valide trouvé.'}
    t_max_global = np.median(t_max_values)

    F_peak_values = [band_results[band]['params']['f_max'] for band in successful_bands if not np.isnan(band_results[band]['params']['f_max'])]
    F_peak_global = np.median(F_peak_values) if F_peak_values else np.nan

    # Calcul des couleurs
    colors = {}
    for i in range(len(BANDS) - 1):
        band1, band2 = BANDS[i], BANDS[i + 1]
        color_key = f'c_{band1}{band2}'
        m_p1 = band_results[band1]['params']['m_p'] if band_results[band1]['success'] else np.nan
        m_p2 = band_results[band2]['params']['m_p'] if band_results[band2]['success'] else np.nan
        if not (np.isnan(m_p1) or np.isnan(m_p2)):
            colors[color_key] = m_p1 - m_p2

    # Extraire les features pour la classification
    features = {}
    for band in BANDS:
        if band_results[band]['success']:
            params = band_results[band]['params']
            features[f't_rise_{band}'] = params['t_rise']
            features[f't_fall_{band}'] = params['t_fall']
        else:
            features[f't_rise_{band}'] = np.nan
            features[f't_fall_{band}'] = np.nan

    return {
        'success': True,
        'global_params': {'t_max': t_max_global, 'F_peak': F_peak_global},
        'band_params': {band: band_results[band] for band in BANDS},
        'colors': colors,
        'features': features,
        'chi2_total': chi2_total,
        'ndof_total': ndof_total,
        'chi2_red': chi2_total / max(ndof_total, 1),
        'error': None
    }

