import numpy as np
import matplotlib.pyplot as plt

logfile = open ('70_driver_log.txt', 'r')

raw_flops = []

for line in logfile.readlines():
    if 'FLOP REPORT' in line:
        raw_flops.append(
            [float(num) for num in line.split(' ')[2:]]
        )

raw_flops = np.array(raw_flops)
print(raw_flops)
print(raw_flops.shape)
training_costs = raw_flops[:, 0]
training_costs /= training_costs[0]
print(training_costs)
epochs = np.arange(0, training_costs.shape[0]) * 10
plt.plot(epochs, training_costs)
plt.xlabel('epoch')
plt.ylabel('% of parameters remaining')

plt.show()
