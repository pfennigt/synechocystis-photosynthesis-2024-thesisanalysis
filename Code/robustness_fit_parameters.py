#!/usr/bin/python3
# %%

import pandas as pd
import numpy as np
import sys
from tqdm.auto import tqdm
import pebble
import pickle
from datetime import datetime
from functools import partial
import warnings
from scipy.optimize import minimize
from concurrent import futures
from pathlib import Path

sys.path.append("../Code")

# Import model functions
from get_current_model import get_model
from function_residuals import calculate_residuals

print(f"Started {datetime.now()}")


# %%
# Set the maximum number of parallel threads and the timeout
# n_workers = 4 # Maximum number of parallel threads
# timeout = 300 # Timeout for each thread in seconds

# Set the prefix to be used for logging and results files
file_prefix = f"minimise_{datetime.now().strftime('%Y%m%d%H%M')}"
# file_prefix = f"residuals_test"


# %%
# Load the model to get default parameter values
m = get_model(get_y0=False, verbose=False, check_consistency=False)

# %%
def get_fitting_startvalue(fitting_parameter, m):
    if fitting_parameter == "fluo_influence":
        return pd.Series({f"__fluo_influence__{k}":v for k,v in m.parameters[fitting_parameter].items()})
    else:
        return pd.Series({fitting_parameter: m.parameters[fitting_parameter]})
    
def get_fitting_bounds(fitting_parameter, bound, m):
    if fitting_parameter == "fluo_influence":
        return pd.Series({f"__fluo_influence__{k}":bound for k in m.parameters[fitting_parameter].keys()})
    else:
        return pd.Series({fitting_parameter: bound})
    
def get_fitting_start_and_bounds(fitting_parameters,m):
    start_values = [get_fitting_startvalue(x,m) for x in fitting_parameters]
    bounds = [get_fitting_bounds(k,v,m) for k,v in fitting_parameters.items()]
    return pd.concat(start_values), pd.concat(bounds).values

def get_fitting_parameter_dict(values, names):
    # Put values into a pandas series for easier handling
    res = pd.Series(values, index=names)

    # Reconstitute fluo_influence
    fluo_influence_bool = res.index.str.startswith("__fluo_influence__")
    if fluo_influence_bool.any():
        # Extract the intermediate values
        _res = res[fluo_influence_bool]
        _res.index = _res.index.str.removeprefix("__fluo_influence__")
        
        res = res[np.invert(fluo_influence_bool)]
        res["fluo_influence"] = _res.to_dict()

    return res.to_dict()

# %%
fitting_parameter_bounds = {
    "fluo_influence": (1e-10,None),
    "lcf": (1e-10,None),
    "kUnquench": (1e-10,None),
    "KMUnquench": (1e-10,None),
    "kQuench": (1e-10,None),
    "kOCPactivation": (1e-10,None),
    "kOCPdeactivation": (1e-10,None),
    "OCPmax": (1e-10,None),
}

start_values, bounds = get_fitting_start_and_bounds(fitting_parameter_bounds,m)

p, p_names = start_values.values, start_values.index

# %%
# Function to calculate residuals with arguments tailored to the minimize function
def calculate_residuals_minimize(p, p_names, scale_factors=None, file_prefix=""):
    # Undo the scaling
    if scale_factors is not None:
        p = p * scale_factors

    _p = get_fitting_parameter_dict(p, p_names)

    res = calculate_residuals(
        _p,
        n_workers=5,
        timeout=300, # s
        logger_filename=f"../out/{file_prefix}",
        save_intermediates=False
        )

    # Save the residuals
    with open(Path(f"../out/{file_prefix}_intermediates.csv",), "a") as f:
        f.writelines(f"{','.join([str(x) for x in p])},{res}\n")

    return res

def scale_bounds(bounds, x0, scale):
    # Scale the vector of bounds according to the scaling of the parameters
    return [tuple([b*(scale/_x0) if b not in [0,None] else b for b in bound]) for bound, _x0 in zip(bounds, x0)]

def fit_model_parameters(start_values, bounds=None, opt_kwargs={}, scale_to_value=None, file_prefix=""):
    
    # If the parameters should be scaled, replace them with the scaling value
    if scale_to_value is None:
        p = start_values.values
        scale_factors = None
    else:
        p = np.full(start_values.shape[0], scale_to_value)
        scale_factors = start_values.values / scale_to_value
        bounds = scale_bounds(bounds, start_values.values, scale_to_value)

    if bounds is not None:
        opt_kwargs.update({"bounds": bounds})

    fit = minimize(
        fun = calculate_residuals_minimize,
        x0 = p,
        args = (start_values.index, scale_factors, file_prefix),
        **opt_kwargs
    )

    # Rescale the results
    if scale_to_value is not None:
        fit.x = fit.x * scale_factors
    return fit

print("Minimising...")


# %%
# Locally optimise the model
with warnings.catch_warnings() as w:
    # Cause all warnings to always be triggered.
    # warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore")
    fit = fit_model_parameters(
        start_values, 
        bounds=bounds, 
        scale_to_value=0.01, 
        opt_kwargs={"method":"Nelder-Mead"},
        file_prefix=file_prefix
        )

# %%
with open(Path(f"../Results/{file_prefix}_results.pickle",), "wb") as f:
    pickle.dump(fit, f)

# with open(Path(f"../Results/{file_prefix}_results.pickle",), "rb") as f:
#     test = pickle.load(f)

# %%
print(fit.message)

print(f"Finished {datetime.now()}")
