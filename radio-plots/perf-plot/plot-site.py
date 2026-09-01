#!/usr/bin/env python

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ["Open Sans"]
plt.rcParams['font.size'] = 12


COLORS = ['#2d3295', '#808080', '#b02020', '#000000']
MARKERS = ['o', 'x', 'D', 's']
LS = ['-'] * 4
LW = [0.5] * 4

WITH_BURNIN=False

x = []
y = []

def getval(part):
    ps = part.split('=')
    return ps[1][:-1]

with open('loop-site-results.txt') as f:
    while True:
        line = f.readline()
        print(line)
        if line == '':
            break
        parts = line.split()
        if len(parts) != 8:
            continue
        print(parts)
        ns = int(getval(parts[3]))
        nst = int(getval(parts[7] + ','))
        t = float(getval(parts[4]))
        if WITH_BURNIN:
            ns -= nw * 10
            print(f'nst: {nst}, ns: {ns}, t: {t}')
        x.append(nst)
        y.append(ns / t)

print(f'x: {x}, y: {y}')

fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(right=0.85)
#ax.set_xscale('log')
ax.set_xticks([4, 8, 16, 32, 64])


baseline = 3.1
baseline_x10 = int(baseline * 10) # we don't want it to show with more precision than the baseline
baseline_st = 1600 / 35 # 1600 samples in 35 seconds
baseline_mt = 1600 / 13
y_max = max(y)
y_ticks = [baseline, baseline_x10, baseline_st, baseline_mt, y_max]

GREEN = '#106000'

for yt in y_ticks:
    if yt == baseline:
        color = COLORS[0]
    elif yt == baseline_mt:
        color = GREEN
    else:
        color = 'black'
    
    plt.axhline(yt, ls=(0, (4, 4)), lw=0.5, color=color, alpha=0.9)

    if yt == baseline_mt:
        ytt = yt + 5
    else:
        ytt = yt
    ax.annotate(f'{yt:.1f}', xy=(-12, yt), xytext=(-50, ytt), xycoords='data', textcoords='data', 
                arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.5}, 
                horizontalalignment='right', verticalalignment='center', color=color)

plt.yticks([])

color = COLORS[0]
ms = MARKERS[0]
ax.plot(x, y, label='x', color=color, marker=ms, ls=LS[0], lw=LW[0], ms=4)

plt.text(600, baseline + 1, '$\\text{RADAR I}$', fontsize=10, color=COLORS[0])
plt.text(600, baseline_st + 1, '$\\text{emcee 1 thread}$', fontsize=10, color='black')
plt.text(600, baseline_mt + 1, '$\\text{emcee multithreaded}$', fontsize=10, color=GREEN)

ax.set_xlabel('$n_{\\text{site threads}}$')
ax.set_ylabel('$\\text{samples} / s$', labelpad=35)
#for spine in ['top', 'bottom', 'right', 'left']:
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(axis='both', which='both', length=0, labelbottom='on', labelleft='on')

#ax.legend()
plt.show()
#plt.savefig('scaling-site.pdf')
