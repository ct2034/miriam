import unittest

import numpy as np

import estimator as e


def generate_data(
        trips_todo: int = 10000,
        timestep: float = .1,
        nr_agents: int = 5,
        nr_landmarks: int = 8
) -> list:
    """
    generating example data that simulates a number of
    agvs
    """
    duration_landmarks_mean = np.arange(4, 4 + nr_landmarks)
    duration_landmarks_std = 1
    assert nr_landmarks == len(duration_landmarks_mean)
    duration_travel_mean = 15
    duration_travel_std = 1

    location = np.zeros(nr_agents)
    timeout = np.zeros(nr_agents)
    _history = []
    t = 0
    while len(_history) < trips_todo:

        at_timeout = timeout <= 0
        at_landmark = location % 1 == 0

        # update agents departing from stations
        for agent in np.arange(nr_agents)[at_landmark & at_timeout]:
            current_landmark = location[agent]
            next_landmark = (current_landmark + 1) % nr_landmarks

            location[agent] = current_landmark + 0.5
            timeout[agent] = np.random.normal(
                loc=duration_travel_mean,
                scale=duration_travel_std
            )

            _history.append([t, current_landmark, next_landmark])

        # update agents arriving at stations
        for agent in np.arange(nr_agents)[np.invert(at_landmark) & at_timeout]:
            arriving_landmark = np.ceil(location[agent]) % nr_landmarks

            location[agent] = arriving_landmark
            timeout[agent] = np.random.normal(
                loc=duration_landmarks_mean[arriving_landmark],
                scale=duration_landmarks_std
            )

        t += timestep
        timeout -= timestep

    return _history


@unittest.skip  # TODO: Finish Tests!
def base_test():
    s = e.init(8)
    e.update(s, 0, 4, 5)
    assert e.estimation(s, 4, 5) == (1, 0), "with only one update we expect a clear result"


@unittest.skip  # TODO: Finish Tests!
def fix_neg_values_test():
    a = np.linspace(-1, 1, 100)
    assert np.min(e.fix_neg_values(a)) == 0, "failed to fix negative values"


@unittest.skip  # TODO: Finish Tests!
def list_test():
    np.random.seed(42)
    n = 3  # number of landmarks
    s = e.init(n)
    l = generate_data(92, .1, nr_agents=4, nr_landmarks=n)
    #  with this setup window is full at iteration 88
    s = e.update_list(s, l)
    e.info(s, plot=True)


if __name__ == "__main__":
    print("main")
    # start = datetime.now()
    # history = generate_data()
    # print("Generated", len(history), "samples")
    # print(history[-5:])
    # print("Computation took", str((datetime.now()-start).total_seconds()), "s")
    list_test()
