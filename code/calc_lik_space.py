import numpy as np
import sys, time, warnings
sys.path.insert(1, './../')
import emuradex
from pyDOE import lhs
from get_inputs import read_input, parse_params
from param_input import params, order
from phase_species import GasPhase, Species

def Nmol_from_dvdr(nh2, dvdr, Xmol, dv=1.0):
    """ Calculate column density of H2 
    nh2 : number density of H2 [cm-3]
    dvdr : velocity gradient [km/s/pc]
    Xmol : molecular abundance wrt H2 density
    dv : line width [km/s]
    """
    Nmol = 3.08e18 * nh2 * dv * Xmol / dvdr
    return Nmol

def get_param_limits(mins, maxs, param_list, par_ID):
    """ 
    Returns
    -------
    par_mis, par_max : dict, inquired 'param_list' parameter limits
    """
    pars = [par for par in param_list if "%s_%i" % (par, par_ID) in mins.keys()]
    par_strings = ["%s_%i" % (par, par_ID) for par in pars]
    par_mins, par_maxs = {}, {}
    for string in par_strings:
        par_mins[string] = mins[string]
        par_maxs[string] = maxs[string]
    return par_mins, par_maxs

def get_param_samples(mins, maxs, rand_samples):
    """ 
    Params
    -------
    rand_samples : array shape=[nsamples,nfeatures], random samples
    Returns
    -------
    params : dict, samples of trial params created in range [mins, maxs]
    """
    param_keys = mins.keys()
    params = {}
    for i, key in enumerate(param_keys):
        params[key.split('_')[0]] = np.power(10, (maxs[key] - mins[key]) * rand_samples[:,i] + mins[key])
    return params

def init_ism(line_table, species_table, species_uniq, params):
    ''' Initialise gas phases and species and first predictions
    Params
    ------
    line_table : pd.DataFrame, observed lines from input.csv
    species_table : list, non-unique species from input.csv table
    species_uniq : list, unique species from input.csv table
    '''
    
    mins, maxs, ps_sets, fixed_params = parse_params(params)
    ps_blocks = []
    loaded_models = {}
    
    physical_params = ['tk', 'f', 'nh2', 'dvdr', 'tbg']

    for phaseID, species in ps_sets.items():
        # load parameter limits for every phase
        phase_mins, phase_maxs = get_param_limits(mins, maxs, physical_params, phaseID)
        Nmol_mins, Nmol_maxs = get_param_limits(mins, maxs, species, phaseID)
        if phaseID == 1:
            # on the first instance create random samples equal for all phases
            nphase_pars = len(list(phase_mins.keys()))
            nspecies = len(list(Nmol_mins))
            ndim = nphase_pars + nspecies
            lh_samples = lhs(ndim, nsamples, criterion='center')
        phase_params = get_param_samples(phase_mins, phase_maxs, lh_samples)
        ps_blocks.append(GasPhase(phaseID, **phase_params))
        for s in species:
            # init species
            Nmol = get_param_samples(Nmol_mins, Nmol_maxs, lh_samples[:,nphase_pars:])
            lineids = line_table['jup'][species_table==s].values
            ps_blocks[-1].add_species(phaseID, s, Nmol[s], lineids, loaded_models)
            models = ps_blocks[-1].species[-1].models
            loaded_models.update(models)
        # first predictions
        ps_blocks[-1].get_fluxes()
        print("phase {} done".format(phaseID))
    return ps_blocks

def make_ratios(ps_blocks, shape, species):
    """ 
    Returns 
    -------
    new_ratios : array, shape=[nsamples, nratios]; ratios of all 
                lines wrt to the first one
    """

    fluxes = np.zeros(shape=shape)
    taus = np.zeros(shape=shape)
    for phase in ps_blocks:
        phase_species = [s.species for s in phase.species]
        ids = np.isin(species, phase_species)
        f_shape = (-1, phase.fluxes.shape[1])
        if np.isscalar(phase.f):
            factors = phase.f
        else:
            factors = np.repeat(phase.f, f_shape[1]).reshape(f_shape)
        fluxes[:,ids] += factors * phase.fluxes
        taus[:,ids] = phase.taus
    
    # create mask for bad tau, flux values
    for i in range(fluxes.shape[1]):
        radex_nan_mask = np.isnan(fluxes[:,i]) + np.isnan(taus[:,i])
        radex_flux_mask = (fluxes[:,i] < 1e-5) | (fluxes[:,i] > flux_cut)
        radex_tau_mask = (taus[:,i] < -1.0) | (taus[:,i] > tau_cut)
    radex_mask = radex_nan_mask + radex_flux_mask + radex_tau_mask
    
    new_ratios = fluxes[~radex_mask, 1:]/fluxes[~radex_mask, 0, None]
    return new_ratios

def make_obs_ratios(lums, d_lums):
    """        
    Params:
    -------
    lums : array, line luminosities from the .csv.
    d_lums : array, uncertainties in the line luminosities from
            the .csv.
    Returns:
    -------
    ratios : array, ratios of line luminosities wrt the first one. 
    """
    
    ratios = []
    er = (d_lums[0] / lums[0])**2
    for lum, d_lum in zip(lums[1:], d_lums[1:]):
        ratio = lum / lums[0]
        error = ratio * np.sqrt((d_lum / lum)**2 + er)
        ratios.append([ratio,error])

    return np.array(ratios)

def chi2(ratios, obs_ratios, obs_errors):
    return np.sum((ratios - obs_ratios[None,:])**2 / obs_errors[None,:]**2, axis=1)

def check_Kvir(ps_blocks):
    """ Check the Kvir condition """
    mask = np.ones(ps_blocks[0].tk.shape[0], dtype=np.bool)
    for phase in ps_blocks:
        Kvir = 1.54 / 1.312 * phase.dvdr / np.sqrt(phase.nh2 / 1000)
        mask *= (Kvir < 0.5) & (Kvir > 20.0)
    return mask


if __name__=='__main__':
    global tau_tol, nsamples
    tau_cut = 100.0
    flux_cut = 200.0 # [K km/s]
    nsamples = np.int32(sys.argv[1])
    
    # log observations
    line_table, species, species_uniq = read_input('input.csv') # log observations
    obs_ratios = make_obs_ratios(line_table['lum'], 
                                 line_table['dlum'])

    t_start = time.time()
    ps_blocks = init_ism(line_table, species, species_uniq, params)
    print("[%.02f sec] ISM initialised" % (time.time() - t_start))
    fluxes_shape = (nsamples, line_table.shape[0])
    trial_ratios = make_ratios(ps_blocks, fluxes_shape, species)

    # check conditions
    mask_kvir = check_Kvir(ps_blocks)
    mask_kvir_ = np.tile(mask_kvir, trial_ratios.shape[1]).reshape(-1, len(mask_kvir)).T
    trial_ratios_msk = np.ma.masked_array(trial_ratios, mask_kvir_)
    
    # calculate chi2 and Lik
    trial_chi2 = chi2(trial_ratios_msk, obs_ratios[:,0], obs_ratios[:,1])
    trial_chi2[mask_kvir] = np.inf
    trial_lik = np.exp(-0.5*(trial_chi2)) # convert chi2 to likelihood
    trial_lik = trial_lik / trial_lik.sum() # normalise likelihood
    print("nsamples", trial_chi2.shape)
    print("non-masked (Kvir) nsamples", trial_chi2[~mask_kvir].shape)
    
    if trial_lik[~mask_kvir] is None:
        warnings.warn("All sampled points were rejected in the process.", 
                      "Please increase nsamples or expand parameter limits.",
                      RuntimeWarning)
    
    if not np.any(trial_lik[~mask_kvir] > 0.0):
        warnings.warn("All sampled points have zero likelihood.",
                      "Try increasing nsamples number or adjusting",
                      "parameter space size and limits.",
                      RuntimeWarning)
    
    # arbitrarily set nonzero cut
    nonzero = trial_lik > 1e-8
    results, header = [], []
    for phase in ps_blocks:
        var_params = phase.get_varParamKeys()
        header.extend(["{}_{}".format(par, phase.ID) for par in var_params])
        for key in var_params:
            results.append(phase.__dict__[key][nonzero])
        for specie in phase.species:
            header.append("%s_%i" % (specie.species, specie.phaseID))
            results.append(specie.Nmol[nonzero])
    header.append("likelihood")
    results.append(trial_lik[nonzero]) # TODO: change to trial_lik
    results = np.array(results).T
    
    # "header.txt" with column headers
    # "lik_space.npy" with columns of params, last col is lik
    np.savetxt("header.txt", header, fmt="%s", delimiter=" ")
    np.save("lik_space.npy", results)
    sys.exit(0)
    