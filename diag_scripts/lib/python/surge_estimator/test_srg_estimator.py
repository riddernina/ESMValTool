import numpy as np
from eofs.standard import Eof 
import sys
import os
import pickle
import ConfigParser
from datetime import datetime
from netCDF4 import Dataset

import surge_estimator
from surge_estimator import surge_estimator_main

#config = ConfigParser.ConfigParser()
#config.read('/usr/people/ridder/Documents/0_models/ESMValTool/nml/cfg_srg_estim/cfg_srg_estim.conf')

# ==================
# I. Load test data
# ==================
print 'Loading test data...'
ECPATH = 'test_data/'

SLPf  = 'daymin_monthly_anom_psl_6h_ECEarth_PD_s01r15_2035'
uf    = 'daymax_monthly_anom_uas_6h_ECEarth_PD_s01r15_2035'
vf    = 'daymax_monthly_anom_vas_6h_ECEarth_PD_s01r15_2035'

nc       = Dataset(ECPATH + SLPf + '.nc')
psl      = nc.variables['var151'][:]
timeSLP  = nc.variables['time'][:]
units    = nc.variables['time'].units
calendar = nc.variables['time'].calendar
nc.close()
#
nc      = Dataset(ECPATH + uf + '.nc')
uas     = nc.variables['var151'][:]
timeu   = nc.variables['time'][:]
nc.close()
#
nc      = Dataset(ECPATH + uf + '.nc')
vas     = nc.variables['var151'][:]
timev   = nc.variables['time'][:]
nc.close()

# ======================================
# II. Call surge estimator main script
# ======================================
print 'Calling surge estimator...'
surge_estimator_main(psl,uas,vas)

