"""
Demonstrates how to schedule a job to be run in a process pool on 3 second intervals.
"""

from datetime import datetime
import time

from apscheduler.schedulers.background import BackgroundScheduler


def tick():
    print("Tick! The time is: %s" % datetime.now())
    print(Test.i)
    Test.i += 1


class Test:
    i = 0


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, "interval", seconds=1)

    Test.i += 1

    try:
        scheduler.start()
        time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        pass
