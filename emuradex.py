import os, glob, joblib
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class Emulator():
    """ Emulator base class """
    def __init__(self, species, trans):
        """
        Input
        -----
        species : str, specie name
        trans : int, Jup quantum number
        """
        self.species = species.upper()
        self.trans = trans
        self.emul_path = os.path.dirname(__file__)
        self.model_dir = self.get_modeldir()
        self.joblib_x = glob.glob(self.model_dir + "/x_transform%s%i*" % (self.species, self.trans))[0]
        self.joblib_y = glob.glob(self.model_dir + "/y_transform%s%i*" % (self.species, self.trans))[0]
        self.x_trans = joblib.load(self.joblib_x)
        self.y_trans = joblib.load(self.joblib_y)
        self.model = tf.keras.models.load_model(self.model_dir)
    
    def get_modeldir(self):
        """ Returns model directory name """
        model_name = "%s-%04d" % (self.species, self.trans)
        model_dir = os.path.join(self.emul_path, "models", self.species, model_name)
        return model_dir

    def call(self, X):
        """ Make inference with emuradex 
        (best used with one sample or <~32 samples)
        """
        X_trans = self.x_trans.transform(X)
        y_trans = self.model(X_trans, training=False) # training off if BatchNorm used
        y = self.y_trans.inverse_transform(y_trans)
        return y

    def predict(self, X, **kwargs):
        """ Make inference with emuradex 
        (best used with larger batches of >~32 samples)
        """
        X_trans = self.x_trans.transform(X)
        y_trans = self.model.predict(X_trans, **kwargs)
        y = self.y_trans.inverse_transform(y_trans)
        return y

# derived class
class Radex(Emulator):
    """ Radex emulator derived class """
    def __init__(self, species, trans):
        Emulator.__init__(self, species, trans)

    def call_flux(self, *args):
        """
        Function to predict fluxes for small input samples
        (1 - 100 samples)

        args:   input to the network (in physical units);
                args shape must be [nsamples, nfeatures];
                features must be the ones network was trained on;
                can be, for eg. temperature, n(H2), n(CO), ...
        preds:  predicted fluxes (in physical units)
        """
        self.check_input(args[0])
        features = np.log10(np.array(args[0])) # features in log10 space
        preds = self.call(features)
        return preds

    def predict_flux(self, *args, **kwargs):
        """
        Function to predict fluxes for large input samples
        (> 100 samples)

        args:   input to the network (in physical units);
                args shape must be [nsamples, nfeatures];
                features must be the ones network was trained on;
                can be, for eg. temperature, n(H2), dvdr, N(CO), ...
        kwargs: tensorflow predict() kwargs
        preds:  predicted fluxes (in physical units) and optical depths 
        """
        self.check_input(args[0])
        features = np.log10(args[0]) # features in log10 space
        self.check_limits(features)
        preds = self.predict(features, **kwargs)
        return preds

    def check_limits(self, features):
        limits = {'tk': [0.5, 3.5], 'nh2': [1, 8], 'dv': [-1.0, 3.0], 
                  'CO': [13, 19], 'CS':[10, 18], 'HCN': [9, 17],
                  'HCO+': [8, 15]}
        current_keys = ['tk', 'nh2', 'dv', self.species]
        for i, key in enumerate(current_keys):
            fmin = np.min(features[:,i])
            fmax = np.max(features[:,i])
            assert (fmin >= np.min(limits[key])) & (fmax <= np.max(limits[key])), "log10(%s)=[%.2f, %.2f] is outside of the emulator range." % (key, fmin, fmax)
        return 0

    def check_input(self, *args):
        if not args:
            raise ValueError("No input data provided to call_flux().",
                             "Provide data X with shape [nsamples, nfeatures].")
        if isinstance(args[0], (list, tuple)):
            raise ValueError("Input X must be a numpy.ndarray or tensorflow tensor.")
        if len(args[0].shape) != 2:
            raise ValueError("Expected 2D array, got %iD array instead." % len(args[0].shape),
                             "Reshape your data either using array.reshape(-1, 1)",
                             "if your data has a single feature or array.reshape(1, -1)" ,
                             "if it contains a single sample.")