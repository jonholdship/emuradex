# Description

This is a neural network emulator of the code Radex for radiative transfer predictions (https://home.strw.leidenuniv.nl/~moldata/radex.html). The model is analogous to the Radex emulator in (https://github.com/drd13/emulchem), only slightly extended and more flexible to use.

Emuradex predicts flux and optical depth, based on four inputs: temperature [K], density of H$_2$ [cm$^{-3}$], line width [km/s] and molecular column density [cm$^{-2}$].

The model here is produced with `tensorflow.keras` (or `tf.keras`) library.

# Install Instructions

Please download the `emuradex` repository to use it. 

In `emuradex/code/` there is `calc_lik_space.py` for making predictions with Radex. The file requires the inputs:  
`input.csv`\
`param_input.py`

The first can be substituted for `input1`.csv` or `input3.csv` that hold data for 1- and 3-phase observations. The `params_input.py` has to be substituted correspondingly for `params_input1.py` or `params_input3.py`. All files are in the same `emuradex/code` directory. Then, run, specifying the number of samples to make predictions for:

`python calc_lik_space.py nsamples`\

# Usage Examples

To start making predictions with the model, you need to do:

`import emuradex`\
`specie = emuradex.Radex("CS", trans=1)`\
`preds = specie.preduct_flux(features)`

where `features.shape` is `(nsamples, 4)` and `preds.shape` is `(nsamples, 2)`




# Useful Notes

1\. The model returned by `tf.keras.models.load_model()` is ready to use.

2\. The `predict()` method (for `tf.keras.Model` object) is optimised for large scale input to work with data batches. Processing small amounts of data (point samples of size that fits into one batch; batch size with which the network was trained can be viewed by calling `model...`; by default predict() batch_size=32) is slow `with predict()` and should instead be done with the `call()` or `__call__()` methods for faster execution (https://www.tensorflow.org/api_docs/python/tf/keras/Model). For example:

- `model(X)`,
- `model.call(tf.convert_to_tensor(X))`,

as opposed to:

- `model.predict(X)`.



## Training a model

The model was trained with the notebook `network_training.ipynb` in Google Colab. The files for training are going to be provided...
