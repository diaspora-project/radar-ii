#!/usr/bin/env python

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ["Open Sans"]
plt.rcParams['font.size'] = 12


COLORS = ['#2d3295', '#808080', '#b02020', '#000000']
MARKERS = ['o', 'x', 'D', 's']
LS = ['-'] * 4
LW = [0.5] * 4

WITH_BURNIN=False

data = {}
NTS = {1, 8, 64, 1024}

def getval(part):
    ps = part.split('=')
    return ps[1][:-1]

with open('loop-results.txt') as f:
    while True:
        line = f.readline()
        print(line)
        if line == '':
            break
        parts = line.split()
        if len(parts) != 7:
            continue
        nw = int(getval(parts[1]))
        nt = int(getval(parts[2]))
        ns = int(getval(parts[3]))
        t = float(getval(parts[4]))
        if WITH_BURNIN:
            ns -= nw * 10
            print(f'nw: {nw}, nt: {nt}, ns: {ns}')
        if nt not in NTS:
            continue
        if nt not in data:
            data[nt] = ([], [])
        nws, y = data[nt]
        nws.append(nw)
        y.append(ns / t)

print(f'Data: {data}')

fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(right=0.85)
#ax.set_xscale('log')
ax.set_xticks([32, 128, 256, 512, 1024])


baseline = data[1][1][0]
baseline_x10 = int(baseline * 10) # we don't want it to show with more precision than the baseline
baseline_st = 1600 / 35 # 1600 samples in 35 seconds
baseline_mt = 1600 / 13
y_max = data[1024][1][-1]
y_ticks = [baseline, baseline_x10, baseline_st, baseline_mt, y_max]

GREEN = '#106000'

for y in y_ticks:
    if y == baseline:
        color = COLORS[0]
    elif y == baseline_mt:
        color = GREEN
    else:
        color = 'black'
    
    plt.axhline(y, ls=(0, (4, 4)), lw=0.5, color=color, alpha=0.9)

    if y == baseline_mt:
        yt = y + 5
    else:
        yt = y
    ax.annotate(f'{y:.1f}', xy=(-12, y), xytext=(-50, yt), xycoords='data', textcoords='data', 
                arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.5}, 
                horizontalalignment='right', verticalalignment='center', color=color)

plt.yticks([])

ix = 0
for nt, (x, y) in data.items():
    if ix < len(COLORS):
        color = COLORS[ix]
    else:
        color = None
    if ix < len(MARKERS):
        ms = MARKERS[ix]
    else:
        ms = None
    ax.plot(x, y, label=str(nt), color=color, marker=ms, ls=LS[ix], lw=LW[ix], ms=4)
    if nt == 1024:
        ypos = y[-1] + 4
    elif nt == 64:
        ypos = y[-1] - 4
    elif nt == 1:
        ypos = y[-1] + 4
    else:
        ypos = y[-1] - 2
    plt.text(x[-1] + 20, ypos, f'$n_t = {nt}$', fontsize=12, color=color)
    ix += 1

plt.text(600, baseline + 1, '$\\text{RADAR I}$', fontsize=10, color=COLORS[0])
plt.text(600, baseline_st + 1, '$\\text{emcee 1 thread}$', fontsize=10, color='black')
plt.text(600, baseline_mt + 1, '$\\text{emcee multithreaded}$', fontsize=10, color=GREEN)

ax.set_xlabel('$n_{\\text{walkers}}$')
ax.set_ylabel('$\\text{samples} / s$', labelpad=35)
#for spine in ['top', 'bottom', 'right', 'left']:
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(axis='both', which='both', length=0, labelbottom='on', labelleft='on')

#ax.legend()
#plt.show()
plt.savefig('scaling.pdf')
