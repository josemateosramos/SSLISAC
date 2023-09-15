# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *

file_name = 'baseline'
if impaired_flag:
    file_name += '_unknown_impairments'
else:
    file_name += 'known_impairments'

"""# Test for a fixed false alarm rate"""

#Test the baseline several times, since results might slightly vary
pd_inter, pfa_inter, rmse_angle_inter, rmse_range_inter, rmse_pos_inter = [], [], [], [], []

for i in range(len(epoch_test_list)):
    pd_temp, pfa_temp, rmse_angle_temp, rmse_range_temp, rmse_pos_temp = \
        testBaselineSensingFixPfa(sigma_sens, theta_min_sens_test, theta_max_sens_test, \
                              range_min_sens_test, range_max_sens_test, K, S, Delta_f, \
                              ant_d, assumed_d, lamb, Ngrid_angle, Ngrid_range, \
                              msg_card, refConst, N0, target_pfa, delta_pfa, \
                              thresholds_pfa, nTestSamples, batch_size, \
                              device)
    pd_inter.append(pd_temp)
    pfa_inter.append(pfa_temp)
    rmse_angle_inter.append(rmse_angle_temp)
    rmse_range_inter.append(rmse_range_temp)
    rmse_pos_inter.append(rmse_pos_temp)

"""# Plot performance"""

plt.figure()
plt.semilogy(epoch_test_list, 1-np.array(np.mean(pd_inter))*np.ones((len(epoch_test_list,))),
             'o--', label='Baseline, ' + ('unknown impairments' if impaired_flag else
                                          'known impairments'))
plt.grid()
plt.xlabel('Number of training iterations')
plt.ylabel('Misdetection probability')
plt.legend()

"""# Save results"""

np.savez(save_path + file_name, \
         pd_inter = pd_inter, pfa_inter = pfa_inter, \
         rmse_angle_inter = rmse_angle_inter, \
         rmse_range_inter = rmse_range_inter, \
         rmse_pos_inter = rmse_pos_inter
        )