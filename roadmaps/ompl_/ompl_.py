try:
    from ompl import base as ob
    from ompl import geometric as og
except (ImportError, ModuleNotFoundError):
    print("OMPL is not installed. Please install OMPL from " +
          "https://ompl.kavrakilab.org/installation.html " +
          "make sure to install the Python bindings. " +
          "`./install-ompl-ubuntu.sh --python`")
    exit(1)
import matplotlib.pyplot as plt
import numpy as np


def isStateValid(state):
    # avoid a circle in the middle of the space
    return True
    # return np.linalg.norm([state.getX(), state.getY()]) > .3


def plan():
    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    si.setStateValidityCheckingResolution(.01)
    si.setup()

    start = ob.State(space)
    start().setX(-.5)
    start().setY(0)

    goal = ob.State(space)
    goal().setX(.5)
    goal().setY(0)

    pd = ob.ProblemDefinition(si)
    pd.setStartAndGoalStates(start, goal)

    # the planner
    p = og.PRM(si)
    # p = og.SPARS(si)
    p.setProblemDefinition(pd)
    # p.setDenseDeltaFraction(1)
    # p.setSparseDeltaFraction(50)
    # p.setStretchFactor(1.01)
    # p.setMaxFailures(500)
    # max_iter = 500000
    # itc = ob.IterationTerminationCondition(max_iter)
    max_time = 5  # stop after 50 seconds
    ptc = ob.timedPlannerTerminationCondition(max_time)
    p.constructRoadmap(ptc)
    # p.printDebug()
    # p.setup()
    p.getRoadmap().printGraphML("roadmap.graphml")

    solved = p.solve(max_time)
    print("Solved:", solved)

    path = []
    if solved:
        ompl_path = p.constructSolution(start, goal)
        for pose in ompl_path.getStates():
            path.append([pose.getX(), pose.getY()])
    path = np.array(path)
    print(path)

    f, ax = plt.subplots()
    ax.plot(path[:, 0], path[:, 1])
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()


if __name__ == "__main__":
    plan()
