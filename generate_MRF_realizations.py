# -*- coding: utf-8 -*-

import os
repo_path_1 = r'D:\00PythonData\pyMRF'
work_path = r'D:\00PythonData\test_data'
os.chdir(work_path)
import numpy as np
import sys
sys.path.append(repo_path_1)
import pyMRF
from matplotlib import pyplot as plt



beta_1 = 5
beta_2 = 0.3
beta_3 = 0.3
beta_4 = 0.3



beta = np.array([beta_1, beta_2, beta_3, beta_4])
num_of_iter=100
       
my_mrf = pyMRF.Element(phys_shp=[100,100], n_labels=3)
my_mrf.fit(beta_prior_mean=beta, fix_beta=True, num_of_iter=num_of_iter)


my_mrf.get_estimator(start_iter=50)
my_mrf.get_label_prob(start_iter=50)
my_mrf.get_map()
my_mrf.get_ie()

labels_bin = my_mrf.labels
beta_bin = np.array(my_mrf.betas)
total_energy = my_mrf.storage_te


MAP_Est = my_mrf.label_map_est.reshape(my_mrf.phys_shp)
Last_realization = my_mrf.labels[-1].reshape(my_mrf.phys_shp)


plt.figure()
#plt.imshow(MAP_Est)
plt.imshow(Last_realization)
plt.xlabel('X Pixle ID', fontsize=14)
plt.ylabel('Y Pixle ID', fontsize=14)
plt.tick_params(labelsize=13)
plt.show()

#np.savetxt('sim_field.csv', Last_realization, delimiter = ',')  


