import estimator as e


def base_test():
    s = e.init(8)
    e.update(s, 0, 4, 5)

    assert e.estimation(s, 4, 5) == (1, 0), "with only one update we expect a clear result"
