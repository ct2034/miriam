from planner.bvc.plan_bvc import plan
from scenarios.visualization import plot_with_paths

if __name__ == "__main__":
    map_img = ((0,),)
    starts = [[0.3, 0.2], [0.6, 0.2], [0.4, 0.8], [0.7, 0.8]]
    goals = [[0.6, 0.8], [0.3, 0.8], [0.8, 0.3], [0.4, 0.3]]

    paths = plan(map_img, starts, goals, radius=0.05)

    print(paths)
    plot_with_paths(map_img, paths)
