import time
from numpy import *

from vfk_msb_py.msb_ws4py_client import MsbWsClient
from vfk_msb_py.msb_classes import *
# from vfk_msb_py.msb_communicate import *

from simulation import SimpSim


s = SimpSim()

def callback(m):
    print(m)

def callback_start(message_dict):
    print("start sim")
    args = message_dict['functionParameters'][0]['value']
    s.start(width=args['x'], height=args['y'], number_agvs=args['n_agvs'])

def callback_job(message_dict):
    args = message_dict['functionParameters'][0]['value']
    print(args)
    s.new_job(
        array([args['start_x'], args['start_y']]),
        array([args['goal_x'], args['goal_y']])
    )

def callback_stop(message_dict):
    print("stop sim")
    s.stop()

if __name__ == '__main__':
    print("__main__.py ...")

    mwc = MsbWsClient('ws://atm.virtualfortknox.de/msb', callback)
    time.sleep(1)

    ePose = Event(
        eventId="Pose",
        name='Pose',
        description='The Pose of an AGV',
        dataFormat=ComplexDataFormat(
            properties=[
                DataFormat("id", "Integer"),
                DataFormat("x", "Integer"),
                DataFormat("y", "Integer")
            ]
        )
    )
    eReached = Event(
        eventId="Reached",
        name="Reached",
        description="An AGV has reached a goal",
        dataFormat=DataFormat(doc_type="Integer")
    )
    fStart = Function(
        functionId="Start",
        name="Start",
        description="Start the Simulation",
        dataFormat=ComplexDataFormat(
            properties=[
                DataFormat("x", "Integer"),
                DataFormat("y", "Integer"),
                DataFormat("n_agvs", "Integer")
            ]
        ),
        callback=callback_start
    )
    fJob = Function(
        functionId="Job",
        name="Job",
        description="Job to Simulate",
        dataFormat=ComplexDataFormat(
            properties=[
                DataFormat("id", "Integer"),
                DataFormat("start_x", "Integer"),
                DataFormat("start_y", "Integer"),
                DataFormat("goal_x", "Integer"),
                DataFormat("goal_y", "Integer")
            ]
        ),
        callback=callback_job
    )
    fStop = Function(
        functionId="Stop",
        name="Stop",
        description="Stop the Simulation",
        callback=callback_stop
    )
    application = Application(
        uuid="3785b920-3777-43ad-9199-b5362d9ef4b5",
        token="b5362d9ef4b5",
        name="AGV sim",
        description="Simulation of AGVs",
        events=[ePose, eReached],
        functions=[fStart, fJob, fStop]
    )

    mwc.register(application)