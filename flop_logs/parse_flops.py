import numpy as np
import matplotlib.pyplot as plt

def plot_flops():
    f1 = 'dynamic.txt'
    f2 = 'otf.txt'
    f3 = 'dynamic-tensor.txt'
    f4 = 'otf-tensor.txt'
    f5 = 'otf-tensor2.txt'

    logfile = open (f5, 'r')

    raw_flops = []

    for line in logfile.readlines():
        if 'FLOP REPORT' in line:
            raw_flops.append(
                [float(num) for num in line.split(' ')[2:]]
            )

    raw_flops = np.array(raw_flops)

    # print(raw_flops)
    # print(raw_flops.shape)

    training_costs = raw_flops[:, 0]
    training_costs /= training_costs[0]
    print(training_costs[-1])
    # exit()
    # print(training_costs)

    epochs = np.arange(0, training_costs.shape[0]) * 10

    plt.ylim((0, 1.1))
    plt.plot(epochs, training_costs)

    plt.xlabel('Epoch')
    plt.ylabel('% FLOPs Compared to Full Network')
    plt.title('OnTheFLy Pruning + Autocast - % of FLOPs Remaining After Each Epoch')

    plt.savefig(f5[:-4] + '.png')
    plt.show()


def plot_times():   
    methods = ['Standard', 'Autocast', 'Dynamic', 'Dynamic + Autocast', 'OnTheFly', 'OnTheFly + Autocast']
    x_nums = np.arange(len(methods))

    times = np.array([1303.163862, 848.080987, 1718.119036, 1773.486697, 1549.297494, 1280.819804]) / 60
    joules = np.array([163.37, 96.01, 147.04, 131.77, 141.53, 105.19])

    fig, axis = plt.subplots(figsize=(10, 4))
    axis2 = axis.twinx()

    width = 0.3

    axis.bar(x_nums - (width/2), times, width=width, color='#1e6ba9')
    axis.set_xticks(x_nums)
    axis.set_xticklabels(methods)

    axis2.bar(x_nums + (width/2), joules, width=width, color='#1aba2a')
    axis2.set_xticks(x_nums)
    axis2.set_xticklabels(methods)

    axis.set_xlabel('Training Method')
    axis.set_ylabel('Training Time (Minutes)')
    axis2.set_ylabel('GPU Energy Usage (Kj)')
    
    plt.title('Total Training Time and GPU Energy Usage per Method')
    plt.legend()
    fig.tight_layout()
    plt.savefig('time_energy.png')
    plt.show()

def plot_energy():

    methods = ['Standard', 'Tensor Core', 'Dynamic', 'Dynamic + Tensor', 'OnTheFly', 'OnTheFly + Tensor']
    joules = np.array([163.37, 96.01, 147.04, 131.77, 141.53, 105.19])

    plt.bar(methods, joules)

    plt.xlabel('Training Method')
    plt.ylabel('GPU Energy Usage (Kj)')
    plt.title('GPU Energy Usage for Each Method')

    plt.savefig('energy.png')
    plt.show()

plot_flops()
# plot_times()
# plot_energy()