# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import random
import sys

from IPython import get_ipython
from matplotlib import pyplot as plt

import scenarios.generators
import solvers
import visualization

cwd = os.getcwd()
assert "miriam" in cwd
sys.path.append(cwd + "/..")


# %%
env, starts, goals = scenarios.generators.tracing_pathes_in_the_dark(
    50, .6, 8, random.Random(0))


# %%
visualization.plot_with_arrows(env, starts, goals)
plt.show()


# %%
paths = solvers.indep(env, starts, goals)


# %%

get_ipython().run_line_magic('matplotlib', 'inline')
visualization.plot_with_paths(env, paths)
plt.show()


# %%
