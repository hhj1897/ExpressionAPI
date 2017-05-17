import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.ndimage.filters import gaussian_filter1d
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", type=str, default='./models/best_no_pain')
parser.add_argument("-s","--sigma", type=float, default=10.)
args = parser.parse_args()



def read_data(path):
    with open(path, 'r') as f:
        content = f.readlines()
        index = content[0].split(',')
        values = []

        for line in content[1:]:
            values.append(line.split(','))

        values = np.array(np.float32(values))
    return values, index


paths = sorted(list(glob(args.input+'/*.csv')))

values, index = read_data(paths[0])

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1,len(index))


color = ['r','g','b','c','m','y','k']
style = {'TR':'--','TE':'-'}

for i, metric in enumerate(index):
    ax = plt.subplot(gs[0, i])
    ax.set_title(metric)
    for n, path in enumerate(paths):
        t = style[path.split('/')[-1][:2]]
        c = color[int(path[-6:-4])]
        values, _ = read_data(path)
        v = values[1:,i]
        epoch = np.arange(v.shape[0])
        v_smooth = gaussian_filter1d(v, args.sigma)
        ax.plot(epoch,v,linestyle=t, color=c,alpha=0.1)
        ax.plot(epoch,v_smooth,linestyle=t, color=c,alpha=1)

plt.show()
