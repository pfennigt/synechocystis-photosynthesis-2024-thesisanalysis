# %%
#!/usr/bin/python3
import pandas as pd
import numpy as np
import sys
from tqdm.auto import tqdm
import pebble
import pickle
from functools import partial
import warnings
import logging
import traceback

from datetime import datetime
from concurrent import futures
from pathlib import Path

sys.path.append("../Code")

# Import model functions
from function_residuals import calculate_residuals, setup_logger
from get_current_model import get_model

# %%
# Set the number of evaluated samples
n_mutations = 10000
include_default_model = True

# Set the maximum number of parallel threads and the timeout
n_workers = 100 # Maximum number of parallel threads
timeout = 500 # Timeout for each thread in seconds

# Set the prefix to be used for logging and results files
file_prefix = f"montecarlo_{datetime.now().strftime('%Y%m%d%H%M')}"
# file_prefix = f"residuals_test"

# Set the random number generator
rng = np.random.default_rng(2024)

# %%
parameter_ranges = {
    # "PSIItot": (0.1,10),
    # "PSItot": (0.1,10),
    # "Q_tot": (0.1,10),
    # "PC_tot": (0.1,10),
    # "Fd_tot": (0.1,10),
    # "NADP_tot": (0.1,10),
    # "NAD_tot": (0.1,10),
    # "AP_tot": (0.1,10),
    # "O2ext": (0.1,10),
    # "bHi": (0.1,10),
    # "bHo": (0.1,10),
    # "cf_lumen": (0.1,10),
    # "cf_cytoplasm": (0.1,10),
    "fCin": (0.1,10),
    # "kH0": (0.1,10),
    # "kHst": (0.1,10),
    # "kF": (0.1,10),
    # "k2": (0.1,10),
    # "kPQred": (0.1,10),
    # "kPCox": (0.1,10),
    # "kFdred": (0.1,10),
    "k_F1": (0.1,10),
    # "k_ox1": (0.1,10),
    "k_Q": (0.1,10),
    # "k_NDH": (0.1,10),
    # "k_SDH": (0.1,10),
    # "k_FN_fwd": (0.1,10),
    # "k_FN_rev": (0.1,10),
    "k_pass": (0.1,10),
    "k_aa": (0.1,10),
    # "kRespiration": (0.1,10),
    # "kO2out": (0.1,10),
    # "kCCM": (0.1,10),
    "fluo_influence": (0.1,10),
    # "PBS_free": (0.1,10),
    # "PBS_PS1": (0.1,10),
    # "PBS_PS2": (0.1,10),
    "lcf": (0.1,10),
    "KMPGA": (0.1,10),
    "kATPsynth": (0.1,10),
    # "Pi_mol": (0.1,10),
    # "HPR": (0.1,10),
    "kATPconsumption": (0.1,10),
    "kNADHconsumption": (0.1,10),
    # "vOxy_max": (0.1,10),
    # "KMATP": (0.1,10),
    # "KMNADPH": (0.1,10),
    # "KMCO2": (0.1,10),
    # "KIO2": (0.1,10),
    # "KMO2": (0.1,10),
    # "KICO2": (0.1,10),
    # "vCBB_max": (0.1,10),
    # "kPR": (0.1,10),
    "kUnquench": (0.1,10),
    "KMUnquench": (0.1,10),
    "kQuench": (0.1,10),
    "KHillFdred": (0.1,10),
    "nHillFdred": (0.1,10),
    # "k_O2": (0.1,10),
    # "cChl": (0.1,10),
    # "CO2ext_pp": (0.1,10),
    # "S": (0.1,10),
    "kCBBactivation": (0.1,10),
    "KMFdred": (0.1,10),
    "kOCPactivation": (0.1,10),
    "kOCPdeactivation": (0.1,10),
    "OCPmax": (0.1,10),
    "vNQ_max": (0.1,10),
    "KMNQ_Qox": (0.1,10),
    "KMNQ_Fdred": (0.1,10),
}

# %%
# Load the model to get default parameter values
m = get_model(get_y0=False, verbose=False, check_consistency=False)

# %%
# Define a function to generate a number of random log-spaced factors to be used with the parameters
def get_mutation_factors(n, start, end, rng):
    rand = rng.random(n)
    return np.exp(np.log(start) + rand*(np.log(end)-np.log(start)))

def get_parameter_mutations(n, parameter_ranges, rng, m=m):
    # Create a container for the mutations
    res = pd.DataFrame(index=np.arange(n), columns=parameter_ranges.keys(), dtype=object)

    for k,v in parameter_ranges.items():
        if k=="fluo_influence":
            _res = {l:m.parameters[k][l] * get_mutation_factors(n, *v, rng=rng) for l in m.parameters[k]}
            res[k] = pd.DataFrame(_res).T.to_dict()
        else:
            # Mutate the default parameter value with the random factors
            res[k] = m.parameters[k] * get_mutation_factors(n, *v, rng=rng)
    return res

# %%
# Define a function to be executed by each thread
def thread_function(x, **kwargs):
    # Unpack the input index and parameter values
    index, p = x

    try:
        # Execute the actual function
        result = calculate_residuals(p, index, **kwargs)

        return index, result

        # Handle the result if needed
    except Exception as e:
        print(f"An error occurred in thread {index} with parameter {p}: {e}")

# %%
# Create the parameters
params = get_parameter_mutations(n_mutations+include_default_model, parameter_ranges, rng, m)

# Include the default model in first position
if include_default_model:
    params.loc[0] = pd.Series({k: m.parameters[k] for k in params.columns})

# Save the parameters
params.to_csv(f"../out/{file_prefix}_params.csv")

# %%
# Create the parameters
params = get_parameter_mutations(n_mutations+include_default_model, parameter_ranges, rng, m)

# Include the default model in first position
if include_default_model:
    params.loc[0] = pd.Series({k: m.parameters[k] for k in params.columns})

# Initialise container for residuals
results = pd.Series(index=np.arange(n_mutations+include_default_model), dtype=float)

if __name__ == "__main__":
    print(f"Started {datetime.now()}")
    print("Monte Carlo simulation...")

    # Setup logging
    InfoLogger = InfoLogger = setup_logger("InfoLogger", Path(f"../out/{file_prefix}_info.log"), level=logging.INFO)
    ErrorLogger = setup_logger("ErrorLogger", Path(f"../out/{file_prefix}_err.log"), level=logging.ERROR)

    # Log the start of the run
    InfoLogger.info("Started run")

    # Catch unnecessary warnings:
    with warnings.catch_warnings() as w:
        # Cause all warnings to always be triggered.
        # warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore")
        
        try:
            # Execute the thread_function for each parameter input
            with tqdm(total=n_mutations+include_default_model) as pbar:
                with pebble.ProcessPool(max_workers=n_workers) as pool:
                    try:
                        for index, res in pool.map(
                            partial(
                                thread_function,
                                intermediate_results_file=f"../out/{file_prefix}_intermediate.csv",
                                logger_filename=f"../out/{file_prefix}",
                            ),
                            params.iterrows(),
                            timeout=timeout,
                        ).result():
                            pbar.update(1)
                            results[index] = res

                    except futures.TimeoutError:
                        pbar.update(1)
                    except Exception as e:
                        pbar.update(1)
                        print(e)
                    finally:
                        pbar.update(1)

            # Save the results
            results.to_csv(f"../out/{file_prefix}_results.csv")
        
            InfoLogger.info(f"Finished run successfully")
        
        except Exception as e:
            ErrorLogger.error("Error encountered\n" + str(traceback.format_exc()))
            InfoLogger.info(f"Finished run with Error")


