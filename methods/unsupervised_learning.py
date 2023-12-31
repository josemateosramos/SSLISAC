# -*- coding: utf-8 -*-
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../lib/')
from simulation_parameters import *

file_name = 'UL'

"""# Training and intermediate testing"""

loss_unsuper, loss_val_super, num_iterations, pd_inter, pfa_inter, \
    rmse_angle_inter, rmse_range_inter, \
    rmse_pos_inter = trainNetwork(network, 0, train_it, sigma_sens, theta_mean_min, theta_mean_max, span_min_theta,
                                  span_max_theta, range_mean_min, range_mean_max, span_min_range,
                                  span_max_range, K, S, Delta_f, lamb, Ngrid_angle,
                                  range_grid, pixels_angle, pixels_range, msg_card, refConst, N0,
                                  ant_d, batch_size, optimizerUnsuper, None, criterionLoss,
                                  epoch_test_list, theta_min_sens_test, theta_max_sens_test,
                                  range_min_sens_test, range_max_sens_test, target_pfa, delta_pfa,
                                  thresholds_pfa, nTestSamples, SL_flag=False, sch_flag=False, device=device)

"""## Plot performance as a function of the number of iterations"""

plt.figure()
plt.semilogy(num_iterations, 1-np.array(pd_inter), 'o-', label='Unsupervised learning (UL)')
plt.grid()
plt.xlabel('Number of training iterations')
plt.ylabel('Misdetection probability')
plt.legend()

"""# Test ISAC"""

pd_isac, pfa_isac, rmse_angle_isac, rmse_range_isac, rmse_pos_isac, ser_isac = \
    testNetworkISAC(sigma_sens, sigma_vector_comm, theta_min_sens_test, theta_max_sens_test, theta_min_comm_test,
                    theta_max_comm_test, range_min_sens_test,
                    range_max_sens_test, Ngrid_angle, range_grid, pixels_angle, pixels_range,
                    K, S, N0, Delta_f, lamb, ant_d, network.d, refConst,
                    rho, phi, target_pfa, delta_pfa, thresholds_pfa, batch_size, nTestSamples, device)

"""### Plot ISAC results"""

plt.figure()
plt.semilogx(ser_isac, 1-np.array(pd_isac), 'o-', label=f'Unsupervised learning (UL)')
plt.grid()
plt.xlabel('Symbol error rate')
plt.ylabel('Misdetection probability')
plt.legend()

plt.figure()
plt.semilogx(ser_isac, rmse_pos_isac, 'o-', label=f'Unsupervised learning (UL)')
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
        num_iterations = num_iterations, \
        loss_unsuper = loss_unsuper, loss_val_super = loss_val_super, \
        pd_isac = pd_isac, pfa_isac = pfa_isac, \
        rmse_angle_isac = rmse_angle_isac, rmse_range_isac = rmse_range_isac, \
        rmse_pos_isac = rmse_pos_isac, ser_isac = ser_isac, \
        learned_d = network.d.cpu().detach().numpy()
        )