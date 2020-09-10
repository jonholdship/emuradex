
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
            'f_min1':-2,
            'f_max1':0,
            'tk_min1':0.5,
            'tk_max1':1.5,
            'nh2_min1':4.0,
            'nh2_max1':8.0,
            #'dvdr_min1':-1.0,
            #'dvdr_max1':3.0,
            'dvdr_1':1.0,
            'hco+_min1':8,#'hco+_min1':-12,
            'hco+_max1':15,#'hco+_max1':-4,
            'hcn_min1':9,#'hcn_min1':-12,
            'hcn_max1':17,#'hcn_max1':-4,
            'co_min1':13,#'co_min1':-8,
            'co_max1':19,#'co_max1':-2,
            
            'f_min2':-2,
            'f_max2':0,
            'tk_min2':2.0,
            'tk_max2':3.5,
            'nh2_min2':1.0,
            'nh2_max2':6.0,
            #'dvdr_min2':-1.0,
            #'dvdr_max2':3.0,
            'dvdr_2':1.0,
            'co_min2':13,#'co_min2':-8,
            'co_max2':19,#'co_max2':-2,
            
            'tk_min3':1.0,
            'tk_max3':3.0,
            'nh2_min3':1.0,
            'nh2_max3':6.0,
            #'dvdr_min3':-1.0,
            #'dvdr_max3':3.0,
            'dvdr_3':1.0,
            'co_min3':13,#'co_min3':-8,
            'co_max3':19,#'co_max3':-2,
            'f_3':0}


"""
List all parameters you wish to vary, including the gas phase as '_X' where 
X = gas phase index (1 based).
"""
order = ['tk_1','nh2_1','dvdr_1','hcn_1','hco+_1','co_1', 'f_1',
         'tk_2','nh2_2','dvdr_2','co_2', 'f_2',
         'tk_3','nh2_3','dvdr_3','co_3']


"""
Optional: give initial values for the varying parameters. Give these in linear 
values (i.e., not log10). 
This can be done as 10**X where X is the log10 value, if necessary.
"""
start_params = {'tk_1':10,
                'nh2_1':1e4,
                'dvdr_1':10,
                'hcn_1':1e-11,
                'hco+_1':1e-8,
                'co_1':1e-5,
                'f_1':0.5,
                'tk_2':500,
                'nh2_2':1e3,
                'dvdr_2':10,
                'co_2':1e-4,
                'f_2':0.5,
                'tk_3':100,
                'nh2_3':1e3,
                'dvdr_3':1,
                'co_3':1e-3}


"""
Enter below the string needed to begin RADEX from the command line.
If RADEX was set up correctly this should not need changing.
"""
radex_call = 'radex'

"""
The maximum optical depth you allow for your lines. van der Tak 2007 recommend 
100.
"""
tau_tol = 1000.


