import numpy as np
import pandas as pd

"""
Set of functions borrowed and slightly from the work by Tunnard et al. (2015)
"""

def read_input(file="input.csv"):
    cols = ['mol', 'jup', 'lum', 'dlum']
    line_table = pd.read_csv(file, sep=',', header=0,
                    usecols=lambda x: x.lower() in cols)
    species = line_table['mol'].values
    species_uniq = np.unique(species)
    return line_table, species, species_uniq


def parse_params(params):
    ''' Process input params.
    
    Params:
    ------
    params: dict, keys in the presribed format and values in log10.
    
    Returns:
    -------
    mins, maxes, ranges and fixed_params : dict.
    '''
    
    n_phases = sum([key[0:6] == 'tk_min' for key in params.keys()])

    min_keys = [key for key in params.keys() if 'min' in key]
    max_keys = [key for key in params.keys() if 'max' in key]
    min_keys.sort() #necessary for range_keys to work properly
    max_keys.sort()
    fixed_keys = [key for key in params.keys() if
                 ('max' not in key and 'min' not in key)]

    assert (len(min_keys) == len(max_keys)), "Different number of mins and maxes! Check param_input.py!!"

    min_vals = [params[min] for min in min_keys]
    max_vals = [params[max] for max in max_keys]
    fixed_vals = [params[key] for key in fixed_keys]
    
    """
    Relabel keys as tk_1, nh2_1 etc
    """
    min_keys = [key[:-4] + key[-1] for key in min_keys]
    max_keys = [key[:-4] + key[-1] for key in max_keys]

    mins = dict(zip(min_keys, min_vals))
    maxes = dict(zip(max_keys, max_vals))
    fixed_params = dict(zip(fixed_keys, fixed_vals))
    
    """
    Identify phase-species pairings
    """
    physical_params = ['tk', 'f', 'nh2', 'dvdr', 'tbg']
    molecules = [mol for mol in params.keys() if
                    not np.any([p in mol for p in physical_params])]
    var_molecules = [m[:-4] + m[-1] if (('min' in m) or ('max' in m))
                    else m for m in molecules]
    var_molecules = np.unique(var_molecules)
    ps_sets = {}
    for ii in range(int(n_phases)):
        ps = [m.split('_')[0] for m in var_molecules if m[-1] == str(ii + 1)]
        ps_sets[ii + 1] = ps
    return (mins, maxes, ps_sets, fixed_params)









