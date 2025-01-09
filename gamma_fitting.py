"""
This mododule is for fitting precipitation data to a hybrid gamma distribution, with related functions,
using xarray data.

The most important function call is
    hybridgamma_best_fit(da_samples, point_dim)
which implements the gamma fitting method F of Appendix A from
Taggart et al (2025), "Ensemble transformations for consistency with a target forecast
and its application to seamless weather to subseasonal forecasting", Bureau Research Report 104.
http://www.bom.gov.au/research/publications/researchreports/BRR-104.pdf

Most important functions are:
    gamma_cdf
    hybridgamma_cdf
    gamma_rvs
    hybridgamma_rvs
    hybridgamma_best_fit
    gamma_fits
    gamma_fit_mse
    hybridgamma_fits
    plot_fits_sample2d
"""

import xarray as xr
import numpy as np
import scipy.stats as ss
import scipy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

def gamma_cdf(x, pars):
    """
    Generates CDF values at points x for gamma distribution with parameters
    loc, scale, shape.
    Args:
        x: xr.DataArray of values at which CDF is evaluated
        pars: xr.Dataset of parameters with names loc, scale and shape.
    Returns:
        xr.DataArray of values        
    """
    pars_bc, x_bc = xr.broadcast(pars, x)
    result = xr.full_like(x_bc, 0)
    result.values = ss.gamma.cdf(x_bc, pars_bc['shape'], loc=pars_bc['loc'], scale=pars_bc['scale'])
    return result


def hybridgamma_cdf(x, pars):
    """
    Generates CDF values at points x for hybrid gamma distribution with parameters
    loc, scale, shape.
    Args:
        x: xr.DataArray of values at which CDF is evaluated
        pars: xr.Dataset of parameters with names loc, scale and shape.
    Returns:
        xr.DataArray of values        
    """    
    gamma_pars = pars.copy()
    gamma_pars['loc'] = gamma_pars['loc'].clip(0)
    pr_dry = (-pars['loc']).clip(0)
    
    gamma_values = gamma_cdf(x, gamma_pars)
    
    result = gamma_values * (1 - pr_dry) + pr_dry
    
    return result.where(x >= 0, 0)
    
def gamma_rvs(pars, sample_size, point_dim):
    """
    Generates sample_size random variates for each entry in pars.
    Returns xr.DataArray with samples enumerated along sample_dim.
    """
    da_samplepoints = xr.DataArray(
        data=np.arange(sample_size),
        dims=[point_dim],
        coords={point_dim: np.arange(sample_size)}
    )
    pars_bc, _ = xr.broadcast(pars, da_samplepoints)
    result = xr.full_like(pars_bc['loc'], 0)
    result.values = ss.gamma.rvs(pars_bc['shape'], loc=pars_bc['loc'], scale=pars_bc['scale'], size=pars_bc['scale'].shape)
    return result.rename('value')

def hybridgamma_rvs(pars, sample_size, point_dim):
    """
    Generates sample_size random variates for each entry in pars.
    Returns xr.DataArray with samples enumerated along sample_dim.
    """
    # if loc >=0, this is just gamma_rvs
    # if loc < 0, this is gamma_rvs with loc=0, then select some variates to be zero
    # with P(dry) = |original_loc|
    
    gamma_pars = pars.clip(0)
    
    result = gamma_rvs(gamma_pars, sample_size, point_dim)
    
    pars_bc, _ = xr.broadcast(pars, result)
    
    p_wet = 1 - (-pars_bc['loc']).clip(0)
    n_binom = xr.ones_like(p_wet).astype(int)
    binomial_multiplier = ss.binom.rvs(n_binom.values, p_wet.values)
    
    return result * binomial_multiplier

def gamma_quantiles(levels, pars):
    """
    Generates quantile values at specified levels for gamma distribution with parameters
    loc, scale, shape.
    Args:
        levels: list, numpy array or xr.DataArray of levels
        pars: xr.Dataset of parameters with names loc, scale and shape.
    Returns:
        xr.DataArray of quantile values
    """
    if not isinstance(levels, xr.core.dataarray.DataArray):
        levels = xr.DataArray(
        data=levels,
        dims=['level'],
        coords=dict(level=levels)
    )
        
    pars1, levels1 = xr.broadcast(pars, levels)

    result = xr.full_like(levels1, np.nan)
    result.values = ss.gamma.ppf(levels1, pars1['shape'], loc=pars1['loc'], scale=pars1['scale'])
    return result

def hybridgamma_quantiles(levels, pars):
    """
    Generates quantile values at specified levels for hybrid gamma distribution with parameters
    loc, scale, shape.
    Args:
        levels: list, numpy array or xr.DataArray of levels
        pars: xr.Dataset of parameters with names loc, scale and shape.
    Returns:
        xr.DataArray of quantile values
    """
    if not isinstance(levels, xr.core.dataarray.DataArray):
        levels = xr.DataArray(
            data=levels,
            dims=['level'],
            coords=dict(level=levels)
        )
    
    # if loc >= 0, same as gama quantiles
    result1 = gamma_quantiles(levels, pars).where(pars['loc'] >= 0, np.nan)
    
    # if loc < 0 and level <= -loc then quantile = 0
    result2 = xr.full_like(result1, 0).where((pars['loc'] < 0) & (levels <= -pars['loc']), np.nan)
    
    # if loc < 0 and level > -loc then
    new_levels = (levels + pars['loc']) / (1 + pars['loc'])
    new_levels = new_levels.where((pars['loc'] < 0) & (levels > -pars['loc']), np.nan)
    new_pars = pars.copy()
    new_pars['loc'] = xr.full_like(new_pars['loc'], 0)
    result3 = gamma_quantiles(new_levels, new_pars)
    
    result = result1.combine_first(result2).combine_first(result3)
    
    return result

def sample_order_statistics(da_samples, point_dim='point', order_dim='order'):
    """Given a data array of samples, returns the order statistics of each sample."""
    
    result = xr.full_like(da_samples, 0).rename({point_dim: order_dim})
    axis_num = da_samples.get_axis_num(point_dim)
    result.values = np.sort(da_samples.values, axis=axis_num)
    result[order_dim] = np.arange(len(result[order_dim]))
    
    return result


def skew(samples, point_dim):
    """
    xarray implementation of scipy.stats.skew with bias=False.
    This implementation handles NaNs in the following way:
        1. If all values are NaN along point_dim, NaN is returned.
        2. Otherwise, NaNs are ignored.
        
    Need to check what happens if sample size <= 2.
    """
    sample_size = samples.count(point_dim)
    sample_mean = samples.mean(point_dim)
    second_moment = ((samples - sample_mean) ** 2).mean(point_dim)
    third_moment = ((samples - sample_mean) ** 3).mean(point_dim)
    result = third_moment / second_moment ** (3 / 2)
    result = np.sqrt(sample_size * (sample_size - 1)) * result / (sample_size - 2)
    return result

def m_fun(z, y, q, point_dim='point'):
    """
    Function used in the BW method.
    z is a float, y is the sample as a data array, q is the exponent.
    """
    return ((y - z) ** q).mean(point_dim)

def match_nans(ds_par):
    """Matches NaNs in parameter dataset."""
    ds_par['loc'] = ds_par['loc'].where(~np.isnan(ds_par['scale']), np.nan).where(~np.isnan(ds_par['shape']), np.nan)
    ds_par['scale'] = ds_par['scale'].where(~np.isnan(ds_par['loc']), np.nan).where(~np.isnan(ds_par['shape']), np.nan)
    ds_par['shape'] = ds_par['shape'].where(~np.isnan(ds_par['scale']), np.nan).where(~np.isnan(ds_par['loc']), np.nan)
    
    return ds_par

def method_mom_findloc(da_samples, point_dim):
    """
    Returns gamma parameters for samples along point_dim,
    based on method of moments. loc is found, not given.
    """
    sample_mean = da_samples.mean(point_dim)
    sample_var = da_samples.var(point_dim)
    sample_skew = skew(da_samples, point_dim)
    
    # parameter estimates
    shape_hat = 4 / sample_skew ** 2
    beta_hat = np.sqrt(shape_hat / sample_var)
    scale_hat = 1 / beta_hat
    loc_hat = sample_mean - shape_hat / beta_hat  
    
    ds_par = [
        loc_hat.rename('loc'),
        scale_hat.rename('scale'),
        shape_hat.rename('shape'),
    ]
    ds_par = xr.merge(ds_par)
    
    return match_nans(ds_par)

def method_bw0_findloc(sorted_sample, order_dim='order'):
    """
    Returns gamma parameters using BW method, loc not given.
    There is no guarantee that loc is positive or less than min(sample).
    Sample size (i.e. length of order_dim) must be at least two, else xr.Dataset()
    is returned.
    """
    if len(sorted_sample[order_dim]) < 2:
        return xr.Dataset()
    
    z1 = 2.2 * sorted_sample.sel({order_dim: 0}) - 1.2 * sorted_sample.sel({order_dim: 1})
    z2 = 3 * sorted_sample.sel({order_dim: 0}) - 2 * sorted_sample.sel({order_dim: 1})
    
    loc_hat = z1.copy()
    
    shape_hat_numerator = (
        - 0.5 * m_fun(z1, sorted_sample, 1, order_dim) * m_fun(z1, sorted_sample, 0.5, order_dim)
        + 0.1 * m_fun(z2, sorted_sample, 0.1, order_dim) * m_fun(z2, sorted_sample, 1.6, order_dim)
        - 0.6 * m_fun(z2, sorted_sample, 1.1, order_dim) * m_fun(z2, sorted_sample, 0.6, order_dim)
    )
    shape_hat_denominator = (
        m_fun(z1, sorted_sample, 1, order_dim) * m_fun(z1, sorted_sample, 0.5, order_dim)
        - m_fun(z1, sorted_sample, 1.5, order_dim)
        + m_fun(z2, sorted_sample, 1.1, order_dim) * m_fun(z2, sorted_sample, 0.6, order_dim)
        - m_fun(z2, sorted_sample, 0.1, order_dim) * m_fun(z2, sorted_sample, 1.6, order_dim)
    ) 
    shape_hat = np.abs(shape_hat_numerator / shape_hat_denominator)
    
    scale_hat = (sorted_sample.mean(order_dim) - loc_hat) / shape_hat
    
    ds_pars = xr.merge([
        loc_hat.rename('loc'), scale_hat.rename('scale'), shape_hat.rename('shape')
    ])
    
    return match_nans(ds_pars)


def method_bw1_findloc(sorted_sample, order_dim='order'):
    """
    Returns gamma parameters using BW NEW-1, including loc.
    There is no guarantee that loc is positive or less than min(sample).
    Sample size (i.e. length of order_dim) must be at least 6, else xr.Dataset()
    is returned.
    """
    if len(sorted_sample[order_dim]) < 6:
        return xr.Dataset()
    
    loc_hat = (
        1.23 * sorted_sample.sel({order_dim: 0}) - 0.23 * sorted_sample.sel({order_dim: 5})
        - 0.22 * (sorted_sample.sel({order_dim: 4}) - sorted_sample.sel({order_dim: 1}))
        - 0.21 * (sorted_sample.sel({order_dim: 3}) - sorted_sample.sel({order_dim: 2}))
    )
    
    ds_pars = bw12p_final(sorted_sample, loc_hat, order_dim)
    
    return match_nans(ds_pars)


def method_bw2_findloc(sorted_sample, order_dim='order'):
    """
    Returns gamma parameters using BW NEW-1, including loc.
    There is no guarantee that loc is positive or less than min(sample).
    Sample size (i.e. length of order_dim) must be at least 3, else xr.Dataset()
    is returned.
    """
    if len(sorted_sample[order_dim]) < 3:
        return xr.Dataset()
    
    loc_hat = (
        2 * sorted_sample.sel({order_dim: 0})
        - 0.5 * (sorted_sample.sel({order_dim: 1}) + sorted_sample.sel({order_dim: 2}))
    )
    
    ds_pars = bw12p_final(sorted_sample, loc_hat, order_dim)
    
    return match_nans(ds_pars)


def bw12p_final(sorted_sample, loc_hat, order_dim):
    """
    Final part of method_locp_bw1 and method_locp_bw2.
    loc_hat could be single float or xr.DataArray.
    """
    scale_hat, shape_hat = bw12_scale_and_shape(sorted_sample, loc_hat, order_dim)
    
    if isinstance(loc_hat, float):
        loc_hat = xr.full_like(scale_hat, loc_hat)

    ds_pars = xr.merge([
        loc_hat.rename('loc'), scale_hat.rename('scale'), shape_hat.rename('shape')
    ])
    
    return ds_pars


def bw12_scale_and_shape(da_samples, loc_hat, point_dim):
    """
    Calculates scale and shape using the BW NEW-1 and NEW-2 methods, given
    loc_hat. Returns scale and shape.
    """
    shape_hat_numerator = -0.4 * m_fun(loc_hat, da_samples, 1, point_dim) * m_fun(loc_hat, da_samples, 0.4, point_dim)
    shape_hat_denominator = (
        m_fun(loc_hat, da_samples, 1, point_dim) * m_fun(loc_hat, da_samples, 0.4, point_dim)
        - m_fun(loc_hat, da_samples, 1.4, point_dim)
    )
    shape_hat = np.abs(shape_hat_numerator / shape_hat_denominator)
    
    scale_hat = (da_samples.mean(point_dim) - loc_hat) / shape_hat

    return scale_hat, shape_hat

def loc_from_sample1(sorted_sample, k, order_dim='order'):
    """
    Returns the minimum of the sample minus delta/2, where
    delta is the smallest positive difference of the lowest k-order stats, or
    the sample minimum, whichever is smaller.
    If k > sample size, then we use the full sample to calculated delta.
    """
    k = min(k, len(sorted_sample[order_dim].values) - 1)
    sample_min = sorted_sample.min(order_dim)
    delta = xr.full_like(sorted_sample.sel({order_dim: list(range(k-1))}), np.nan)
    
    axis_num = sorted_sample.get_axis_num(order_dim)
    delta.values = np.diff(sorted_sample.sel({order_dim: list(range(k))}), axis=axis_num)
    delta = delta.where(delta > 0, np.nan)
    delta = delta.min(order_dim) 
    delta = delta.where(delta < sample_min, sample_min)
    
    return sample_min - delta / 2


def loc_from_sample2(sorted_sample, a=2, b=2.5, order_dim='order'):
    """
    Returns y1 - a(y2 - y1) - b(y3 - y2) as an estimate of the loc
    parameter when positive, else y1 / 2. Here 0 < y1 <= y2 <= y3 are the smallest three values from the sample.
    Designed for cases when the sample has no zeros.
    """
    y1 = sorted_sample.sel({order_dim: 0})
    loc_hat = (
        y1
        - a * (sorted_sample.sel({order_dim: 1}) - sorted_sample.sel({order_dim: 0}))
        - b * (sorted_sample.sel({order_dim: 2}) - sorted_sample.sel({order_dim: 1}))
    )
    loc_hat = loc_hat.where(loc_hat > 0, y1 / 2)
    
    return loc_hat


def method_mom_loc0(da_samples, point_dim):
    """
    Calculates scale and shape parameters for gamma distribution given loc_hat=0,
    using the method of moments.
    """
    sample_mean = da_samples.mean(point_dim)
    sample_variance = da_samples.var(point_dim)
    
    scale_hat = sample_variance / sample_mean
    shape_hat = sample_mean ** 2 / sample_variance
    loc_hat = xr.zeros_like(scale_hat)
    
    ds_pars = [
        loc_hat.rename('loc'),
        scale_hat.rename('scale'),
        shape_hat.rename('shape'),
    ]
    ds_pars = xr.merge(ds_pars)
    
    return match_nans(ds_pars)


def method_mom_locgiven(sample, point_dim, loc):
    """
    Uses the method of moments to calculate gamma parameters, given loc.
    """
    sample = sample - loc
    ds_pars = method_mom_loc0(sample, point_dim)
    ds_pars['loc'] = ds_pars['loc'] + loc
    
    return match_nans(ds_pars)


def method_bw12_locgiven(sorted_sample, order_dim, loc):
    """
    Uses the BW NEW-1,2 method to calculate gamma parameters, given loc.
    """
    ds_pars = bw12p_final(sorted_sample, loc, order_dim)
    
    return match_nans(ds_pars)


def method_bw12_loc0(sorted_sample, order_dim):
    """
    Uses the BW NEW-1,2 method to calculate gamma parameters, given loc.
    """
    ds_pars = bw12p_final(sorted_sample, 0., order_dim)
    
    return match_nans(ds_pars)


def method_expontial_loc0(da_samples, point_dim='point'):
    """
    mean of the nonzero sample is an unbiased estimator of the scale.
    loc = 0, shape = 1, scale = mean(nonzero points in sample)
    """
    scale = da_samples.where(da_samples > 0, np.nan).mean(point_dim).rename('scale')
    loc = xr.full_like(scale, 0).rename('loc')
    shape = xr.full_like(scale, 1.).rename('shape')
    pars = xr.merge([loc, scale, shape])
    return match_nans(pars)
    
# used for gamma fits only
METHODS = ['mom_findloc', 'bw0_findloc', 'bw1_findloc', 'bw2_findloc',
           'bw_loc0', 'mom_loc0',
           'mom_locgiven1', 'bw_locgiven1', 'mom_locgiven2', 'bw_locgiven2',
          ]
# used when loc=0 has an assumption for gamma fit
METHODS_LOC0 = ['bw_loc0', 'mom_loc0']

# used for hybrid gamma fits
EDGE_METHODS = ['exponential', 'all_zeros']
ALL_METHODS = METHODS + EDGE_METHODS

RECOMMENDED_METHODS = ['bw_loc0', 'mom_loc0', 'mom_locgiven1', 'bw_locgiven1']


METHOD_DICT = dict(zip(ALL_METHODS, list(range(len(ALL_METHODS)))))
    
def gamma_fits(da_samples, point_dim='point', methods=METHODS, k=6, a=2, b=2.5):
    """
    Gets gamma fits for samples using various methods. NaNs are ignored.
    """
    sorted_sample = sample_order_statistics(da_samples, point_dim)
    sample_count = sorted_sample.count('order')
    
    pars = []
    
    if 'mom_findloc' in methods:
        pars.append(
            method_mom_findloc(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['mom_findloc'])
            .expand_dims('method_num')
        )
    if 'bw0_findloc' in methods:
        pars.append(
            method_bw0_findloc(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['bw0_findloc'])
            .expand_dims('method_num')
        )
    if 'bw1_findloc' in methods:
        pars.append(
            method_bw1_findloc(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['bw1_findloc'])
            .expand_dims('method_num')
        )
    if 'bw2_findloc' in methods:
        pars.append(
            method_bw2_findloc(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['bw2_findloc'])
            .expand_dims('method_num')
        )
    if 'bw_loc0' in methods:
        pars.append(
            method_bw12_loc0(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['bw_loc0'])
            .expand_dims('method_num')
        )
    if 'mom_loc0' in methods:
        pars.append(
            method_mom_loc0(sorted_sample, 'order')
            .assign_coords(method_num=METHOD_DICT['mom_loc0'])
            .expand_dims('method_num')
        )
        
    if 'mom_locgiven1' or 'bw_locgiven1' in methods:
        alt_loc1 = loc_from_sample1(sorted_sample, k, 'order')
    if 'mom_locgiven1' in methods:
        pars.append(
            method_mom_locgiven(sorted_sample, 'order', alt_loc1)
            .assign_coords(method_num=METHOD_DICT['mom_locgiven1'])
            .expand_dims('method_num')
        )
    if 'bw_locgiven1' in methods:
        pars.append(
            method_bw12_locgiven(sorted_sample, 'order', alt_loc1)
            .assign_coords(method_num=METHOD_DICT['bw_locgiven1'])
            .expand_dims('method_num')
        )
        
    if 'mom_locgiven2' or 'bw_locgiven2' in methods:
        alt_loc2 = loc_from_sample2(sorted_sample, a, b, 'order')
    if 'mom_locgiven2' in methods:
        pars.append(
            method_mom_locgiven(sorted_sample, 'order', alt_loc2)
            .assign_coords(method_num=METHOD_DICT['mom_locgiven2'])
            .expand_dims('method_num')
        )
    if 'bw_locgiven2' in methods:
        pars.append(
            method_bw12_locgiven(sorted_sample, 'order', alt_loc2)
            .assign_coords(method_num=METHOD_DICT['bw_locgiven2'])
            .expand_dims('method_num')
        )
        
    # we NaN any fit where there is only one nonzero value
    pars = xr.merge(pars)
    pars = pars.where(sample_count > 1)
    
    # this provides a fit whem there is only one nonzero value
        
    pars_exp = (
        method_expontial_loc0(sorted_sample, 'order')
        .assign_coords(method_num=METHOD_DICT['exponential'])
        .expand_dims('method_num')
    )
    
                
    pars = xr.merge([pars, pars_exp])    
    mse_of_fits = gamma_fit_mse(sorted_sample, pars, 'order')
    result = xr.merge([pars, mse_of_fits])
    
    if 'order' in result.coords:
        result = result.drop('order')
        
    return result


def gamma_fit_mse(sorted_sample, ds_pars, order_dim='order'):
    """
    Returns mse of gamma fits
    """    
    cdf_values = gamma_cdf(sorted_sample, ds_pars)
    
    length_of_sample = sorted_sample.count(order_dim)
        
    error = cdf_values - (sorted_sample[order_dim] + 1) / (length_of_sample + 1)
    
    return (error ** 2).mean(order_dim).rename('mse')

# def hybridgamma_fit_mse(sorted_sample, ds_pars, order_dim='order'):
#     """
#     Returns mse of hybrid gamma fits
#     """    
#     cdf_values = hybridgamma_cdf(sorted_sample, ds_pars)
    
#     length_of_sample = sorted_sample.count(order_dim)
#     zeros_in_sample = (sorted_sample == 0).sum(order_dim)
        
#     error = cdf_values - (sorted_sample[order_dim] + 1) / (length_of_sample + 1)
#     error = error.where(sorted_sample != 0, np.nan)
    
#     result = (error ** 2).mean(order_dim).rename('mse')
#     result = result.where(length_of_sample != zeros_in_sample, 0).where(length_of_sample > 0, np.nan)
    
#     return result


def hybridgamma_fits(da_samples, point_dim='point', methods=METHODS, k=6, a=2, b=2.5, mse_include_zero=True):
    """
    Returns hybrid gamma fits and mses in one dataset.
    If `mse_include_zero`, zero forecasts are included in the MSE calculation as a zero error.
    """
    
    # count the zeros in each sample
    zero_count = (da_samples == 0).sum(point_dim)
    sample_length = da_samples.count(point_dim)
    
    # NOTE: may need to introduce more cases, based on number of zeros
    # zero points are NaNed
    samples_with_zeros = da_samples.where(zero_count > 0, np.nan).where(da_samples != 0, np.nan)
    samples_without_zeros = da_samples.where(zero_count == 0, np.nan)
    
    # samples with zeros have gamma loc = 0
    meths = list(set(methods).intersection(set(METHODS_LOC0)))
    fits1 = gamma_fits(samples_with_zeros, point_dim, meths, k=k, a=a, b=b)
    # update hybrid gamma loc so that it reflects the proportion of zeros
    fits1['loc'] = -zero_count / sample_length
    fits1 = match_nans(fits1)
    
    fits1['mse'] = fits1['mse'].where(~np.isnan(fits1['loc']), np.nan)
    
    # samples without zeros
    fits2 = gamma_fits(samples_without_zeros, point_dim, methods, k=k, a=a, b=b)
    # only keep fits where loc >= 0 and loc < min(sample)
    fits2['loc'] = (
        fits2['loc']
        .where(fits2['loc'] >= 0., np.nan)
        .where(fits2['loc'] < da_samples.min(point_dim), np.nan)
    )
    fits2 = match_nans(fits2)
    fits2['mse'] = fits2['mse'].where(~np.isnan(fits2['loc']), np.nan)
    
    # edge cases
    fits3 = hybridgamma_edgecase_fits(da_samples, point_dim)
        
        
    fits = xr.merge([fits1, fits2, fits3])
    
    # correct MSEs if zeros included
    if mse_include_zero:
        fits['mse'] = fits['mse'] * (sample_length - zero_count) / sample_length
    
    return fits


def hybridgamma_best_fit(da_samples, point_dim='point', methods=RECOMMENDED_METHODS, k=6, a=2, b=2.5):
    """
    Returns parameters and mse for the best hybrid gamma fit.
    If data variable 'member_num' has a value of -1, this is equivalent to no method.
    """
    
    fits = hybridgamma_fits(da_samples, point_dim=point_dim, methods=methods, k=k, a=a, b=b)
    
    # need to remove NaNs for argmin to work, (all NaN slice encountered)
    # replace NaNs with IMPOSSIBLE_MSE > 1, since 0 <= mse <= 1
    IMPOSSIBLE_MSE = 2.
    mses = fits['mse'].where(~np.isnan(fits['mse']), IMPOSSIBLE_MSE)
    
    best_mses = mses.min('method_num')
    best_method_nums = mses.idxmin('method_num').rename('method_num')
    best_fits = fits.where(fits['method_num'] == best_method_nums).min('method_num')
    result = xr.merge([best_fits, best_method_nums]).astype(float)
    
    result = result.where(result['mse'] != IMPOSSIBLE_MSE, np.nan)
    
    # now need to convert method_num back to int
    # np.nan needs to be replaced by integer equivalent of NAN. 
    INT_NAN = -1
    result['method_num'] = result['method_num'].where(~np.isnan(result['loc']), np.nan)
    result['method_num'] = result['method_num'].where(~np.isnan(result['method_num']), INT_NAN)
    result['method_num'] = result['method_num'].astype(int)
    
    return result


def hybridgamma_edgecase_fits(da_samples, point_dim='point'):
    """
    Tackles edge cases. Returns fits and mses.
    The only edge case implemented here is  where all non-NaN data points are 0.
    """
    
    # where there is at least one and only zeros in the sample
    # return loc=-1, shape=scale=1
    
    sample_max = da_samples.max(point_dim)
    loc0 = xr.full_like(sample_max, -1.).rename('loc')
    scale0 = xr.full_like(sample_max, 1.).rename('scale')
    shape0 = xr.full_like(sample_max, 1.).rename('shape')
    mses = xr.full_like(sample_max, 0.).rename('mse')
    fits = xr.merge([loc0, scale0, shape0, mses])
    
    # only keep these values where sample max is 0
    fits = (
        fits
        .where(sample_max == 0)
        .assign_coords(method_num=METHOD_DICT['all_zeros'])
        .expand_dims('method_num')
    )
    
    return fits


def plot_fits_sample2d(da_samples, fits, point_dim = 'point'):
    """
    Returns a plotly figure object with fitted hybrid gamma cdf and emprical quantiles.
    Assumes two dimensions to specify sample (e.g. date, station)
    """
    
    sample_shape = da_samples.max(point_dim).shape
    sample_dims = da_samples.max(point_dim).dims
    
    fig = make_subplots(*sample_shape)

    xvals = np.linspace(-2, 1.1 * float(da_samples.max()), 2000)
    da_xvals = xr.DataArray(
        data=xvals,
        dims=['x'],
        coords=dict(x=xvals)
    )
    fits = hybridgamma_best_fit(da_samples)
    yvals = hybridgamma_cdf(da_xvals, fits)
    sorted_sample = sample_order_statistics(da_samples)


    for i in range(sample_shape[0]):
        for j in range(sample_shape[1]):

            fig.add_trace(go.Scatter(
                x=xvals, 
                y=yvals.isel({sample_dims[0]: i, sample_dims[1]: j}),
                mode='lines',
                #line=dict(color=line_colours[0], width=line_size),
                name='original CDF',
            ), i + 1, j + 1)

            this_sorted_sample = sorted_sample.isel({sample_dims[0]: i, sample_dims[1]: j})
            this_sorted_sample = this_sorted_sample[~np.isnan(this_sorted_sample)]
            this_sample_size = len(this_sorted_sample)

            fig.add_trace(go.Scatter(
                x=this_sorted_sample, 
                y=(np.arange(this_sample_size) + 1) / (this_sample_size + 1),
                mode='markers',
                #line=dict(color=line_colours[0], width=line_size),
                name='sample points',
            ), i + 1, j + 1)

            fig.update_layout(yaxis_range=[-.1,1.1])

    fig.update_layout(
        autosize=False,
        width=900,
        height=1200,
    )
    
    return fig