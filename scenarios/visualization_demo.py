# %%
import os
import sys

import scenarios.generators
import solvers
import visualization

cwd = os.getcwd()
assert "miriam" in cwd
sys.path.append(cwd + "/..")


# %%
env, starts, goals = scenarios.generators.tracing_pathes_in_the_dark(
    50, .6, 8, 0)


# %%
visualization.plot_with_arrows(env, starts, goals)


# %%
paths = solvers.indep(env, starts, goals)


# %%
visualization.plot_with_paths(env, paths)
