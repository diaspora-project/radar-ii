#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys
import math


FONT_SZ = 12


plt.rcParams['font.sans-serif'] = ["Open Sans"]
plt.rcParams['font.size'] = FONT_SZ


_COLORS = ['black', 'black', 'black', '#D72000', '#1BB6AF', 'black', '#FFAD0A', '#9093A2', '#EE6100', '#132157', 'black', 'black']
LINE_HEIGHT = 0.15
BAR_HEIGHT = 0.11

X_MIN = 99.888


class Thread():
    def __init__(self, ttid):
        self.ttid = ttid
        self.stime = None
        self.times = []

    def start(self, time):
        self.stime = time

    def end(self, time) -> float:
        self.times.append((self.stime, time - self.stime))
        return time - self.stime

    def plot(self, ax, index, y):
        ax.broken_barh(self.times, (y * LINE_HEIGHT, BAR_HEIGHT), color=_COLORS[index], 
                       ec='#000000', lw=0.5)

class Counter:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.threads = {}
        self.n = 0
        self.s = 0.0
        self.ssq = 0.0

    def start(self, pid, tid, time):
        ttid = f'{pid}-{tid}'
        if ttid not in self.threads:
            self.threads[ttid] = Thread(ttid)
        self.threads[ttid].start(time)

    def end(self, pid, tid, time):
        ttid = f'{pid}-{tid}'
        try:
            d = self.threads[ttid].end(time)
            self.n += 1
            self.s += d
            self.ssq += d * d
        except KeyError:
            pass

    def avg(self) -> float:
        return self.s / self.n

    def stdev(self) -> float:
        return math.sqrt(self.ssq / self.n - (self.s / self.n) ** 2)

    def plot(self, ax, cid, pos):
        for tid, thread in self.threads.items():
            if tid in pos:
                y = pos[tid]
            else:
                y = len(pos)
                pos[tid] = y

            thread.plot(ax, cid, y)

counters = {}


t_0 = None

with open('timers-combined.dat') as f:
    for line in f:
        line = line.strip()
        els = line.split()
        if line[0] == '<':
            cid = els[1]
            pid = els[2]
            tid = els[3]
            time = float(els[4])
            counters[cid].end(pid, tid, time - t_0)
        elif line[0] == '>':
            cid = els[1]
            pid = els[2]
            tid = els[3]
            time = float(els[4])
            if t_0 is None:
                t_0 = time
            rel_t = time - t_0
            counters[cid].start(pid, tid, rel_t)
        else:
            id = els[0]
            name = els[1]
            counters[id] = Counter(name, id)

cix = 0

n_threads = 0
threads = set()
for counter in counters.values():
    threads.update(counter.threads.keys())
n_threads = max(len(threads), len(counters) * 2)


fig, ax = plt.subplots(figsize=(11,  n_threads * LINE_HEIGHT))
plt.xlim(X_MIN, 101.1)
pos = {}
lines = []
labels = []

NICE_NAMES = {
    '136679711954592': 'Server send',
    '136679711961552': 'Server wait for results',
    '126794255490784': 'Site send',
    '126794235138352': 'afterglowpy'
}

total_time = 0.0
total_var = 0.0

for cid, counter in counters.items():
    counter.plot(ax, cix, pos)
    if cid in NICE_NAMES:
        label = f'$\\text{{{NICE_NAMES[cid]}:}}\\, {counter.avg() * 1000:.1f} \\pm {counter.stdev() * 1000:.1f} ms$'
        if 'Server' in label:
            total_time += counter.avg()
            total_var += counter.stdev() ** 2
        labels.append(label)
        lines.append(Line2D([0], [0], color=_COLORS[cix], lw=8))
    cix += 1

#total_lines = [Line2D([0], [0], color='white', lw=8)]
#total_labels = [f'$\\text{{Iteration total:}}\\, {total_time * 1000:.1f} \\pm {math.sqrt(total_var) * 1000:.1f} ms$']
lines += [Line2D([0], [0], color='white', lw=8)]
labels += [f'$\\text{{Iteration total:}}\\, {total_time * 1000:.1f} \\pm {math.sqrt(total_var) * 1000:.1f} ms$']

ly = []
ll = ['Site', 'Server']

for tid, y in pos.items():
    ly.append(y * LINE_HEIGHT + (LINE_HEIGHT - BAR_HEIGHT / 3) / 2)

ax.set_yticks(ly, ll)
ax.tick_params(axis='y', pad=10)
ax.invert_yaxis()
ax.set_xlabel('$\\text{time (s)}$')

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

x_ticks = [X_MIN, X_MIN + 0.25, X_MIN + 0.365, X_MIN + 0.5, X_MIN + 0.75, X_MIN + 1]

for x in x_ticks:
    plt.axvline(x=x, color='black', ls='dotted', alpha=0.5)

ax.set_xticks(x_ticks, labels=[f'${x - X_MIN:.2f}$' for x in x_ticks])

l1 = ax.legend(lines, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.00), bbox_transform=fig.transFigure,
               fontsize=FONT_SZ, alignment='center', frameon=False)
plt.gca().add_artist(l1)
#l2 = ax.legend(total_lines, total_labels, loc='lower center', ncol=1,
#               bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure,
#               fontsize=FONT_SZ, alignment='center', frameon=False)
plt.subplots_adjust(left=0.14, right=0.95, bottom=0.4)
ax.tick_params(axis='both', which='both', length=0, labelbottom='on', labelleft='on')
plt.savefig(f'profile.pdf')
#plt.show()
