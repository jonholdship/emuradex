
"""
List all fixed params and upper and lower limits on the parameter ranges.
Values are expected to be given as log10.
Order is not important, as this is a dictionary and hashed.

parameters:
tk   [Kelvin]
nh2  [cm^-3]
dvdr [km s^-1 pc^-1]
Tbg  [Kelvin]
mol  [Abundance relative to H2] -- give this in the same form as the RADEX
                                   molecular data file.
"""

params = {  'tbg_1':3,
            'f_1':1,
            'tk_min1':0.5,
            'tk_max1':1.5,
            'nh2_min1':4.0,
            'nh2_max1':8.0,
            #'dvdr_min1':-1.0,
            #'dvdr_max1':3.0,
            'dvdr_1':1000.0,
            'hco+_min1':8,
            'hco+_max1':15,
            'hcn_min1':9,
            'hcn_max1':17
            }

"""
List all parameters you wish to vary, including the gas phase as '_X' where 
X = gas phase index (1 based).
"""
order = ['tk_1','nh2_1','dvdr_1','hcn_1','hco+_1']


"""
The maximum optical depth you allow for your lines. van der Tak 2007 recommend 
100.
"""
tau_tol = 100.


