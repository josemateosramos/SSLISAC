# -*- coding: utf-8 -*-
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *

file_name = 'baseline'
if impaired_flag:
    file_name += '_unknown_impairments'
else:
    file_name += '_known_impairments'

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

"""# Test ISAC"""

pd_isac, pfa_isac, rmse_angle_isac, rmse_range_isac, rmse_pos_isac, ser_isac = \
    testBaselineISAC(sigma_sens, sigma_vector_comm, theta_min_sens_test, theta_max_sens_test, theta_min_comm_test,
                    theta_max_comm_test, range_min_sens_test,
                    range_max_sens_test, Ngrid_angle, Ngrid_range,
                    K, S, N0, Delta_f, lamb, true_d, assumed_d, refConst,
                    rho, phi, target_pfa, delta_pfa, thresholds, batch_size, nTestSamples, device='cpu')

"""### Plot ISAC results"""

plt.figure()
plt.semilogx(ser_isac, 1-np.array(pd_isac), 'o-', label=f'Supervised learning (SL)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Misdetection probability')
plt.legend()

plt.figure()
plt.semilogx(ser_isac, rmse_pos_isac, 'o-', label=f'Supervised learning (SL)')
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
        ser_isac = ser_isac, pd_isac = pd_isac, \
        pfa_isac = pfa_isac, rmse_angle_isac = rmse_angle_isac, \
        rmse_range_isac = rmse_range_isac, rmse_pos_isac = rmse_pos_isac
        )