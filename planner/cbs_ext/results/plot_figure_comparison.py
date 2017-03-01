import datetime

import numpy as np
import pickle

from planner.cbs_ext.plan import plan
import os

from planner.cbs_ext.planner_test import get_data_random
import random
import itertools
import matplotlib.pyplot as plt

plt.style.use('bmh')

# data from process_test.py - test_benchmark() @ tag:iros17_comparison
dat = [128.016001, 81.024077]

ypos = [0, .5]

plt.figure(figsize=[8, 5])
plt.barh(ypos, dat, height=.4)
plt.xlabel('Production Time [s]')
# plt.ylabel('Planning Time Saving [%]')
ax = plt.gca()
ax.set_yticks(ypos)
ax.set_yticklabels(['Nearest', 'CBSEXT'])
ax.set_ylim(bottom=-.4, top=.9)

# plt.show()
plt.savefig('figure_comparison.png', bbox_inches='tight')
