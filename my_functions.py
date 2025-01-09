"""
Functions used to reproduce results in Taggart & Wilke, "Warnings based on risk matrices: a coherent framework with
consistent evaluation"
"""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import shapefile as shp


def risk_matrix_score(fcst, obs, weights, prob_thresholds, dim_sev_threshold):
    """
    Calculates the risk matrix score for each forecast case.
    
    Args:
        fcst: xr.DataArray of probability exceedance forecasts for each severity threshold.
        obs: xr.DataArray of binary observations (1 if severity threshold exceeded, 0 otherwise).
        weights: xr.DataArray of weights to put on each sev_thresholds and prob_thresholds
        prob_thresholds: list of probability thresholds
        dim_sev_threshold: name (str) of the severity threshold dimension

    Returns:
        xarray object with the risk matrix scores.
    """
    da_prob_thresholds = xr.DataArray(
        data=prob_thresholds,
        dims=['prob_threshold'],
        coords={'prob_threshold': prob_thresholds}
    )

    mask = (~np.isnan(fcst)) & (~np.isnan(obs))

    over = da_prob_thresholds.where((obs == 0) & (fcst > da_prob_thresholds), 0).where(mask)
    under = (1 - da_prob_thresholds).where((obs == 1) & (fcst <= da_prob_thresholds), 0).where(mask)
    result = (over + under) * weights
    result = result.sum(['prob_threshold', dim_sev_threshold], skipna=False)
    
    return result


def wt_matrix_to_xr(wt_matrix, prob_thresholds, sev_thresholds, sev_thresholds_descending=False):
    """
    Converts a matrix of weights to an xarray data array, designed to be used as the
    `weights` argument in `risk__matrix_score`.

    Args:
        wt_matrix: 2D numpy matrix of weights.
            Rows (top to bottom) correspond to weights for probability thresholds (decending).
            Columns (left to right) correspond to weights for severity thresholds (with increasing severity).
        prob_thresholds: list of probability thresholds.
        sev_thresholds: list of severity thresholds.
        sev_thresholds_descending: If true, severity increases with decreasing threshold value.
            If false, it is assumed that severity increases with increasing threshold value.

    Returns:
        xr.DataArray of weights with correct coordinates.
    """
    sev_coords = np.sort(np.array(sev_thresholds))
    if sev_thresholds_descending:
        sev_coords = np.flip(sev_coords)
        
    prob_coords = np.flip(np.sort(np.array(prob_thresholds)))  # must be descending
    da_weights = xr.DataArray(
        data=wt_matrix,
        dims=['prob_threshold', 'sev_threshold'],
        coords={
            'prob_threshold': prob_coords,
            'sev_threshold': sev_coords
        }
    )
    
    return da_weights

# precipitation processing
def clip_small_amounts(data, clip_threshold=0.2):
    """
    Returns data, with amounts less than clip_threshold set to 0.
    """
    return data.where(data >= clip_threshold, 0.).where(~np.isnan(data), np.nan)


# functions for converting lat/lon to albers equal area projection

# projection information for native IMPROVER projection
SEMI_MAJOR_AXIS =  6378137.0
SEMI_MINOR_AXIS =  6356752.314140356
REF_LONGITUDE = 132.0
REF_LATITUDE = 0.
STANDARD_PARALLELS = [-18., -36.]


def deg_to_rad(angle):
    """Convert from degrees to radians."""
    return angle * np.pi / 180

def latlon_to_albers(lat, lon, ref_lon=REF_LONGITUDE, ref_lat=REF_LATITUDE, std_parallels=STANDARD_PARALLELS):
    """
    Returns the x and y coordinates of Albers equal area projection, given lat and lon coordinates.

    Args:
        lat: numpy array of latitude values, in degrees
        lon: corresponding numpy array of longitude values, in degrees
        ref_lon (float): reference longitude for Albers equal area projection, in degrees
        ref_lat (float): reference latitude for Albers equal area projection, in degrees
        std_parallels (list of two floats): standard parallels for Albers equal area projection, in degrees

    Returns
        tuple of two elements: (1) a numpy arrray of Albers equal area x projection coordinates;
        (2) a corresponding numpy arrray of Albers equal area y projection coordinates.
    """
    # convert to radians
    phi1 = deg_to_rad(STANDARD_PARALLELS[0])
    phi2 = deg_to_rad(STANDARD_PARALLELS[1])
    lat = deg_to_rad(lat)
    lon = deg_to_rad(lon)
    lon0 = deg_to_rad(REF_LONGITUDE)
    lat0 = deg_to_rad(REF_LATITUDE)

    # the following formula based on https://en.wikipedia.org/wiki/Albers_projection,
    # with adjustments for ellipsoidal model of earth.
    n = 0.5 * (np.sin(phi1) + np.sin(phi2))
    theta = n * (lon - lon0)
    C = np.cos(phi1) ** 2 + 2 * n * np.sin(phi1)
    rho = SEMI_MAJOR_AXIS * np.sqrt(C - 2 * n * np.sin(lat)) / n
    rho0 = SEMI_MINOR_AXIS * np.sqrt(C - 2 * n * np.sin(lat0)) / n
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)

    return x, y


def add_public_wx_districts(sf, color):
    """
    Adds public weather district boundaries to a matplotlib plot in the native IMPROVER grid.
    sf is the shape file with the public weather districts.
    `color` specifies the colour of the boundaries that are plotted.
    """
    for shape in sf.shapeRecords():
        for i in range(len(shape.shape.parts)):
            i_start = shape.shape.parts[i]
            if i==len(shape.shape.parts)-1:
                i_end = len(shape.shape.points)
            else:
                i_end = shape.shape.parts[i+1]
            y = [i[0] for i in shape.shape.points[i_start:i_end]]
            x = [i[1] for i in shape.shape.points[i_start:i_end]]
            x, y = latlon_to_albers(np.array(x), np.array(y))
            plt.plot(x, y, color=color, linewidth=.5)


# helper functions for annotating plots in Albers equal area projection
def get_pos(rel_pos, lims):
    """
    Get the projection ordinate given relative position on the plot. Lims is a tuple: usually XLIMS or YLIMS
    """
    return rel_pos * lims[1] + (1 - rel_pos) * lims[0]

def get_len(rel_len, lims):
    """
    Get the length in projection space given relative length on the plot. Lims is a tuple: usually XLIMS or YLIMS
    """
    return rel_len * (lims[1] - lims[0])