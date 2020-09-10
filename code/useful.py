import numpy as np
import sys

#freq, flux = sys.argv[1:]
#freq = np.float32(freq)
#flux = np.float32(flux)

def flux_to_lum(freq, flux):
    # freq [GHz]
    d = 107 # Mpc
    z = 0.0245
    ang_scale = 0.494 # kpc/arcsec
    return 3.25 * 10**7 * freq**(-2) * d**2 * (1 + z)**(-3) * flux * 1.05

#print("L=%.2e (K km/s pc2)" % flux_to_lum(freq, flux))