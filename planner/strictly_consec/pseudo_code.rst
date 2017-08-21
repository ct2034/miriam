idea
----

we optimize sum of all trips between tasks and tasks themselves

free_agents = agents
free_tasks = tasks
consec = {}
agent_task = {}

while len(free_tasks > 0):
	poses = if len(free_tasks) > len(free_agents) : free_agents & task_ends : free_agents
	closest_pose, closest_task = nearest_neighbor(poses, free_tasks) // as of path (only eval 2 best candidates or something)
	free_tasks -= closest_task
	if type(closest_pose) == task
		consec[task] = closest_pose // (task) 
	else: // is an agent
		agent_task[task] = closest_pose
		free_agents -= closest_pose // as agent
