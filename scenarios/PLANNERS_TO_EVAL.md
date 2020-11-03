## Hierarchical cooperative A* (HCA*)

    Silver, D. (2005). Cooperative Pathfinding. AIIDE, 117–122. http://www.aaai.org/Library/AIIDE/aiide05contents.php


## Increasing cost tree search (ICTS)

    Shi, R., Steenkiste, P., & Veloso, M. M. (2019). SC-M*: A multi-agent path planning algorithm with soft-collision constraint on allocation of common resources. Applied Sciences (Switzerland), 9(19). https://doi.org/10.3390/app9194037

* Search in a meta tree of cost sets (cost per agent)
* In low-level search per-agent paths with that fixed cost
* _Assumptions_
    * Agents at goal occupy that space until al agents are finished
    * Collisions can ocurr on nodes and edges
* _Cost_
    * __Sum of costs__ "The summation (over all agents) of the number of time steps required to reach the goal location."
    * __Makespan__, which is the total time until
the last agent reaches its destination (i.e., the maximum of the individual costs).

