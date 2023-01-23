import ompl
import ompl.base as ob
import ompl.geometric as og

# create an SE2 state space
space = ob.SE2StateSpace()

# set lower and upper bounds
bounds = ob.RealVectorBounds(2)
bounds.setLow(-1)
bounds.setHigh(1)
space.setBounds(bounds)

# create a space information instance
si = ob.SpaceInformation(space)

# define a state validity checker


def isStateValid(state):
    # implement your state validity checker here
    return True


si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
si.setStateValidityCheckingResolution(.01)
si.setup()

# create a start and goal state
start = ob.State(space)
goal = ob.State(space)

# set start and goal state values
start[0] = 0
start[1] = 0
start[2] = 0
goal[0] = 1
goal[1] = 1
goal[2] = 0

# create a problem instance
pdef = ob.ProblemDefinition(si)
pdef.setStartAndGoalStates(start, goal)

# create a planner
planner = og.SPARS(si)

# set planner parameters
planner.setDenseDeltaFraction(1)
planner.setSparseDeltaFraction(50)
planner.setStretchFactor(1.01)
planner.setMaxFailures(500)
planner.setProblemDefinition(pdef)

# set termination condition
max_time = 60  # stop after 60 seconds
ptc = ob.timedPlannerTerminationCondition(max_time)

# solve the problem
solved = planner.solve(max_time)

if solved:
    # print the path to the console
    print(pdef.getSolutionPath())
else:
    print("No solution found")
