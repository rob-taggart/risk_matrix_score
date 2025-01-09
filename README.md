# risk_matrix_score

This repository contains all of the data and most of the python code used in the paper "Warnings based on risk matrices: a coherent framework with consistent evaluation" by Robert J. Taggart and David J. Wilke (2025).

This readme file gives an overview of repository contents and their relationship to the paper.

Throughout, data is stored and manipultated using the xarray package.

*my_functions.py*

my_functions.py is a python module consisting of functions that are called in the notebooks below. It contains an implementation of the risk matrix score. Users are encouraged to use the python `scores` package implementation of risk_matrix_score, which has more features. The implementation here is sufficient to reporoduce results of the paper.

*gamma_fitting.py*

gamma_fitting.py is a python module consisting of functions that are used to fit non-negative, potentially zero-inflated, data to a gamma distribution (or mixture of gamma and Bernoulli distributions, in the case of zero inflation). The gamma fitting method used is described in Appendix A of Taggart et al (2025) http://www.bom.gov.au/research/publications/researchreports/BRR-104.pdf

*synthetic_experiment.ipynb*

synthetic_experiment.ipynb is a python notebook that is used to generate the synthetic results of Section 3.4, and in particular the results of Table 6. 

*plots_tcj_spatial.ipynb*

plots_tcj_spatial.ipynb is a notebook used to generate Figures 8 (panel of warnings) and 9 (map of observed rainfall) for the Tropical Cyclone Jasper example of Section 4.

*score_plots.ipynb*

score_plots.ipynb is a notebook used to generate Figure 10 (mean scores for different forecast systems for the the Tropical Cyclone Jasper case study).

**data**

The data directory conists of the following datasets:

- aeps_at_stations.nc: annual exceedance probability (AEP) thresholds for 24-hour duration at the 177 rain gauge locations (indexed by station number) used in Section 4. AEPs thresholds are indexed by the percentage chance of exceedance in any given year.
- aeps_gridded.nc: annual exceedance probability (AEP) thresholds for 24-hour duration on a grid across northern Queensland used in Section 4. AEPs thresholds are indexed by the percentage chance of exceedance in any given year. The grid is an Albers equal area projection, and this projection is shared by other gridded datasets in this directory.
- gridded_precipitation24h_bom_access_ge3_20231217.nc: gridded forecast 24-hour precipitation values across the north Queensland domain from the ACCESS-GE3 ensemble for 17 December 2023.
- gridded_precipitation24h_ecmwf_ens_20231217.nc: gridded forecast 24-hour precipitation values across the north Queensland domain from the ECMWF ensemble for 17 December 2023.
- gridded_precipitation24h_ecmwf_hres_20231217.nc: gridded forecast 24-hour precipitation values across the north Queensland domain from the ECMWF deterministic model for 17 December 2023.
- observations.csv: 24-hour rainfall accumulations observed at at the 177 rain gauge locations (indexed by station number) for 17 December 2023. Station metadata is included.
- gfe_shape_files: shape files for the public weather districts and coastline used in Figure 8.
- station_precipitation24h_bom_access_ge3_20231217.nc: forecast 24-hour precipitation values at the 177 rain gauge locations (indexed by station number) from the ACCESS-GE3 ensemble for 17 December 2023.
- station_precipitation24h_ecmwf_ens_20231217.nc: forecast 24-hour precipitation values at the 177 rain gauge locations (indexed by station number) from the ECMWF ensemble for 17 December 2023.
- station_precipitation24h_ecmwf_hres_20231217.nc: forecast 24-hour precipitation values at the 177 rain gauge locations (indexed by station number) from the ECMWF deterministic model for 17 December 2023.
