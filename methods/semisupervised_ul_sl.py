# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *

file_name = 'SSL_UL_SL'

"""# Training

## Unsupervised training and intermediate testing
"""

loss_unsuper, loss_val_super, num_it_unsuper, pd_inter_unsuper, pfa_inter_unsuper, \
    rmse_angle_inter_unsuper, rmse_range_inter_unsuper, \
    rmse_pos_inter_unsuper = trainNetwork(network, 0, train_it_unsuper, sigma_sens, theta_mean_min, theta_mean_max, span_min_theta,
                                  span_max_theta, range_mean_min, range_mean_max, span_min_range,
                                  span_max_range, K, S, Delta_f, lamb, Ngrid_angle,
                                  range_grid, pixels_angle, pixels_range, msg_card, refConst, N0,
                                  ant_d, batch_size, optimizerUnsuper, None, criterionLoss,
                                  epoch_test_list, theta_min_sens_test, theta_max_sens_test,
                                  range_min_sens_test, range_max_sens_test, target_pfa, delta_pfa,
                                  thresholds_pfa, nTestSamples, SL_flag=False, sch_flag=False, device=device)

"""## Supervised training and intermediate testing"""

loss_super, _, num_it_super, pd_inter_super, pfa_inter_super, \
    rmse_angle_inter_super, rmse_range_inter_super, \
    rmse_pos_inter_super = trainNetwork(network, train_it_unsuper, train_it, sigma_sens, theta_mean_min, theta_mean_max, span_min_theta,
                                  span_max_theta, range_mean_min, range_mean_max, span_min_range,
                                  span_max_range, K, S, Delta_f, lamb, Ngrid_angle,
                                  range_grid, pixels_angle, pixels_range, msg_card, refConst, N0,
                                  ant_d, batch_size, optimizerSuper, scheduler_super, criterionLoss,
                                  epoch_test_list, theta_min_sens_test, theta_max_sens_test,
                                  range_min_sens_test, range_max_sens_test, target_pfa, delta_pfa,
                                  thresholds_pfa, nTestSamples, SL_flag=True, sch_flag=True, device=device)

"""## Plot performance as a function of the number of iterations"""

#Concatenate supervised and unsupervised results
pd_inter = np.concatenate((pd_inter_unsuper, pd_inter_super))
pfa_inter = np.concatenate((pfa_inter_unsuper, pfa_inter_super))
rmse_angle_inter = np.concatenate((rmse_angle_inter_unsuper, rmse_angle_inter_super))
rmse_range_inter = np.concatenate((rmse_range_inter_unsuper, rmse_range_inter_super))
rmse_pos_inter = np.concatenate((rmse_pos_inter_unsuper, rmse_pos_inter_super))
num_iterations = np.concatenate((num_it_unsuper, num_it_super))

plt.figure()
plt.semilogy(num_iterations, 1-np.array(pd_inter), 'o-', label=f'SSL: UL+SL({train_it_super} iterations)')
plt.grid()
plt.xlabel('Number of training iterations')
plt.ylabel('Misdetection probability')
plt.legend()

"""# Test ISAC"""

pd_isac, pfa_isac, rmse_angle_isac, rmse_range_isac, rmse_pos_isac, ser_isac = \
    testNetworkISAC(sigma_sens, sigma_vector_comm, theta_min_sens_test, theta_max_sens_test, theta_min_comm_test,
                    theta_max_comm_test, range_min_sens_test,
                    range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                    K, S, N0, Delta_f, lamb, true_d, network.d, refConst,
                    rho, phi, target_pfa, delta_pfa, thresholds_pfa, batch_size, nTestSamples, device)

"""### Plot ISAC results"""

plt.figure()
plt.semilogx(ser_isac, 1-np.array(pd_isac), 'o-', label=f'SSL: UL+SL({train_it_super} iterations)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Misdetection probability')
plt.legend()

plt.figure()
plt.semilogx(ser_isac, rmse_pos_isac, 'o-', label=f'SSL: UL+SL({train_it_super} iterations)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Position RMSE [m]')
plt.legend()

"""# Save results"""

np.savez(save_path + file_name, \
        pd_inter = pd_inter, pfa_inter = pfa_inter, \
        rmse_angle_inter = rmse_angle_inter, \
        rmse_range_inter = rmse_range_inter, \
        rmse_pos_inter = rmse_pos_inter, \
        num_iterations = num_iterations, loss_super = loss_super, \
        loss_unsuper = loss_unsuper, loss_val_super = loss_val_super, \
        pd_isac = pd_isac, pfa_isac = pfa_isac, \
        rmse_angle_isac = rmse_angle_isac, rmse_range_isac = rmse_range_isac, \
        rmse_pos_isac = rmse_pos_isac, \
        learned_d = network.d.cpu().detach().numpy()
        )