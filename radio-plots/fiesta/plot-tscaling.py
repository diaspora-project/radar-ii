#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ["Open Sans"]
plt.rcParams['font.size'] = 12


COLORS = ['#2d3295', '#808080', '#b02020', '#000000']
MARKERS = ['o', 'x', 'D', 's']
LS = ['-'] * 4
LW = [0.5] * 4

data = {}
NTS = {1, 8, 64, 1024}

def getval(part):
    ps = part.split('=')
    if ps[1][-1] == ',':
        return ps[1][:-1]
    else:
        return ps[1]


with open('data.txt') as f:
    while True:
        line = f.readline()
        print(line)
        if line == '':
            break
        parts = line.split()
        if len(parts) != 5:
            continue
        k = getval(parts[0])
        nw = int(getval(parts[1]))
        nt = int(getval(parts[2]))
        nit = int(getval(parts[3]))
        t = float(getval(parts[4]))
        #if nt not in NTS:
        #    continue
        if k not in data:
            data[k] = {}
        if nt not in data[k]:
            data[k][nt] = ([], [])
        nws, y = data[k][nt]
        nws.append(nw)
        ns = 200 * nw
        y.append(ns / t)

print(f'Data: {data}')


fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(right=0.85, bottom=0.15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks([2, 8, 32, 128, 512, 2048])
ax.set_yticks([1, 5, 10, 100, 500, 1000, 5000, 10000, 50000, 100000])
plt.ylim([10, 5000])


GREEN = '#106000'


#plt.yticks([])

def getxy(datav):
    # all entries for which nw == nt
    xl = []
    yl = []
    
    for nt, (nws, ys) in datav.items():
        for ix in range(len(nws)):
            nw = nws[ix]
            y = ys[ix]
            if nw == 32:
                xl.append(nt)
                yl.append(y)
            elif nt == nw:
                xl.append(nw)
                yl.append(y)

    xl = np.array(xl)
    yl = np.array(yl)
    si = np.argsort(xl)
    xl = xl[si]
    yl = yl[si]
    return xl, yl



LINES = [('gpu', 'FiestaEM GPU'), ('cpu', 'FiestaEM CPU'), ('af', 'Afterglowpy')]

ix = 0
for p in LINES:
    key, label = p
    datav = data[key]
    x, y = getxy(datav)

    print(f'x: {x}\ny: {y}')
    if ix < len(COLORS):
        color = COLORS[ix]
    else:
        color = None
    if ix < len(MARKERS):
        ms = MARKERS[ix]
    else:
        ms = None
    ax.plot(x, y, label=label, color=color, marker=ms, ls=LS[ix], lw=LW[ix], ms=4)
    print(f'ix: {ix}')
    if ix == 2:
        x_label = x[-1] - 1000
        y_label = y[-1] * 1.1
    else:
        x_label = x[-1] + 20
        y_label = y[-1] * 0.93
    plt.text(x_label, y_label, f'$\\text{{{label}}}$', fontsize=12, color=color)
    ix += 1


ax.set_xlabel('$n_{t}$', labelpad=0)
ax.set_ylabel('$\\text{samples} / s$', labelpad=0)
#for spine in ['top', 'bottom', 'right', 'left']:
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

#ax.tick_params(axis='both', which='both', length=0, labelbottom='on', labelleft='on')

#ax.legend()
#plt.show()
plt.savefig('fiesta.pdf')
