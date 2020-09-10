import numpy as np
import sys
sys.path.insert(1, '/mnt/c/Users/Vadim/google_drive/colab-notebooks/tf_logs/saved_models/')
import emuradex

class Species:
    def __init__(self, phaseID, species, Nmol, lineids, loaded_models):
        self.phaseID = phaseID
        self.species = species
        self.Nmol = Nmol
        self.lineids = lineids.tolist() # VR
        for i, val in enumerate(self.lineids):  # VR
            self.lineids[i] = int(val)  # VR
        self.models = {}
        self.load_emulator(loaded_models)
        
    def load_emulator(self, loaded_models):
        """ Load models on object initialisation. 
            
            Models are loaded once for repeated calls
            to the same models.
        """
        for i in self.lineids:
            key = "{}{}".format(self.species, i)
            if key in loaded_models.keys():
                self.models[key] = loaded_models[key]
            else:
                self.models[key] = emuradex.Radex(self.species, i)
        
    def get_fluxes(self, phase_params, **kwargs):
        """ Returns inferences (fluxes, taus) using the loaded models """
        line_js = self.lineids
        ps_params = np.concatenate((phase_params, self.Nmol.reshape(-1,1)), axis=1)
        fluxes, taus = [], []
        for j_up in line_js:
            key = "{}{}".format(self.species, j_up)
            pred = self.models[key].predict_flux(ps_params)
            flux, tau = pred[:,0], pred[:,1]
            fluxes.append(flux)
            taus.append(tau)
        return fluxes, taus


class GasPhase:
    def __init__(self, ID, tk, nh2, dvdr=1000.0, tbg=2.73, dv=10.0, f=1.0):
        self.ID = ID
        self.species = []
        
        self.physical_params = ['tk', 'nh2', 'dvdr', 'tbg', 'f']
        self.tk = tk
        self.nh2 = nh2
        self.dvdr = dvdr
        self.tbg = tbg
        self.f = f

        self.dv = dv
        self.Nh2 = None
        if np.size(self.dv) == np.size(self.tk):
            pass
        else:
            self.dv = np.full(self.tk.shape, dv, dtype=np.float32)
        self.fluxes, self.taus = [], []

    def get_varParamKeys(self):
        """ Returns non-fixed pysical parameters """
        params_keys = []
        for key in self.physical_params:
            if np.size(self.__dict__[key]) > 1:
                params_keys.append(key)
        return params_keys
    
    def get_varParamLimits(self, which=None, cf=0.99):
        """ Takes 'which' indices.
        
            Returns a reduced parameter space, defined by 
            confidense regions of the likelihood.
        """
        var_params = self.get_varParamKeys()
        params_mins = {}
        params_maxs = {}
        bins = 20
        for key in var_params:
            param = self.__dict__[key][which]
            counts, bin_edges = np.histogram(param, bins=bins)
            level = self.get_cfs(counts, cf)
            bw1 = bin_edges[:-1]
            bw2 = bin_edges[1:]
            param_min = bw1[counts>=level].min()
            param_max = bw2[counts>=level].max()
            params_mins[key] = np.log10(param_min)
            params_maxs[key] = np.log10(param_max)

        for specie in self.species:
            param = specie.Nmol[which]
            counts, bin_edges = np.histogram(param, bins=bins)
            level = self.get_cfs(counts, cf)
            bw1 = bin_edges[:-1]
            bw2 = bin_edges[1:]
            param_min = bw1[counts>=level].min()
            param_max = bw2[counts>=level].max()
            params_mins[specie.species] = np.log10(param_min)
            params_maxs[specie.species] = np.log10(param_max)
        return params_mins, params_maxs
    
    def get_cfs(self, counts, cf=0.99):
        ''' Find the counts that correspond to the confidence intervals provided.

        Params:
        ------
        counts: array, list; binned data from plt.hist function or np.histogram (or 2d).
        cf : scalar, the confidence intervals requested -  e.g., [0.68,0.95,0.99]
            
        Returns:
        -------
        levels : scalar, the count values corresponding to the requested confidence regions
        '''
        
        points = np.copy(np.sort(counts,axis=None))[::-1]
        total = points.sum()
        cumsum = np.cumsum(points)
        point = points[cumsum>=cf*total]
        if len(point) > 1:
            point = points[cumsum>=cf*total][0]
        return point
        
    def add_species(self, phaseID, species, Nmol, lineids, loaded_models):
        """ Add species object to the Phase object """
        self.species.append(Species(phaseID, species, Nmol, lineids, loaded_models))

    def get_params(self):
        """ Returns: physical features for emuradex predictions """
        params = np.array([self.tk, self.nh2, self.dv]).T
        return params
    
    def get_fluxes(self, **kwargs):
        """ call Specie get_fluxes() to store fluxes, taus """
        phase_params = self.get_params()
        self.fluxes = []
        for specie in self.species:
            fluxes, taus = specie.get_fluxes(phase_params)
            self.fluxes.extend(fluxes)
            self.taus.extend(taus)
        self.fluxes = np.array(self.fluxes).T
        self.taus = np.array(self.taus).T
        