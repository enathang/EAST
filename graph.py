import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

# Code from Yaya

#data = np.loadtxt('run_K40_Dataset-tag-global_step_sec.csv', delimiter=',', skiprows=1)

#reader_data = np.loadtxt('run_K40_Reader-tag-global_step_sec.csv', delimiter=',', skiprows=1)

#data = np.loadtxt('run_K80+K80_Dataset-tag-global_step_sec.csv', delimiter=',', skiprows=1)
#reader_data = np.loadtxt('run_K80+K80_Reader-tag-global_step_sec.csv', delimiter=',', skiprows=1)


'''data = np.loadtxt('run_K80_default_pinning_Dataset_5c5a33d-tag-global_step_sec.csv', delimiter=',', skiprows=1)
reader_data = np.loadtxt('run_K80+K80_cpu_bucketbatch_pinning_Dataset_5c5a33d-tag-global_step_sec.csv', delimiter=',', skiprows=1)'''


data_4 = np.loadtxt('run_data_model1_48-tag-loss.csv', delimiter=',', skiprows=1)
data_5 = np.loadtxt('run_data_model2_48-tag-loss.csv', delimiter=',', skiprows=1)
data_6 = np.loadtxt('run_data_model3_48-tag-loss.csv', delimiter=',', skiprows=1)


'''data_4 = np.loadtxt('run_data_model1_48-tag-global_step_sec.csv', delimiter=',', skiprows=1)
data_5 = np.loadtxt('run_data_model2_48-tag-global_step_sec.csv', delimiter=',', skiprows=1)
data_6 = np.loadtxt('run_data_model3_48-tag-global_step_sec.csv', delimiter=',', skiprows=1)'''


time_4 = data_4[:,0]
step_4 = data_4[:,1]
value_4 = data_4[:,2]


time_5 = data_5[:,0]
step_5 = data_5[:,1]
value_5 = data_5[:,2]

time_6 = data_6[:,0]
step_6 = data_6[:,1]
value_6 = data_6[:,2]



def smooth(step, alpha):
    last = step[0]
    new_step = list()
    for val in step:
        smoothed_val = last * alpha + (1 - alpha) * val
        new_step.append(smoothed_val)
        last = smoothed_val

    return new_step

batch_4 = smooth(value_4, 0.9)
batch_5 = smooth(value_5, 0.9)
batch_6 = smooth(value_6, 0.9)

#Reader vs DS
'''plt.plot(step, modified, linewidth=2.0, label = 'New pipeline')
plt.plot(step, modified_read, linewidth=2.0, label = 'Old pipeline')
plt.legend(loc='lower right')
plt.xlabel('Total steps taken', fontsize=10)
plt.ylabel('Global steps/sec', fontsize=10)
plt.suptitle('Throughput:  Old pipeline vs New pipeline when 2 GPUs are exposed', fontsize=10, ha='center')
plt.show()'''


#Pinning expts
'''plt.plot(step, modified, linewidth=2.0, label = 'Default pinning')
plt.plot(step, modified_read, linewidth=2.0, label = 'Pinning to the CPU')
plt.legend(loc='lower right')
plt.xlabel('Total steps taken', fontsize=10)
plt.ylabel('Global steps/sec', fontsize=10)
plt.suptitle('Throughput: Default pinning of bucketbatch op vs pinning to the CPU', fontsize=10, ha='center')
plt.show()'''

#batch level results
plt.plot(step_4, batch_4, linewidth=2.0, label = 'Tower 1')
plt.plot(step_5, batch_5, linewidth=2.0, label = 'Tower 2')
plt.plot(step_6, batch_6, linewidth=2.0, label = 'Tower 3')
plt.legend(loc='upper right')
plt.xlabel('Total steps taken', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.suptitle('Loss convergence of different towers', fontsize=10, ha='center')
plt.show()
