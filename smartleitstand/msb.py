import time
from numpy import *
from PyQt4 import QtGui, QtCore

from vfk_msb_py.msb_ws4py_client import MsbWsClient
from vfk_msb_py.msb_classes import *
# from vfk_msb_py.msb_communicate import *


def callback(m):
    if "NIO" in str(m):
        print(m)


def callback_start(message_dict):
    print("start sim")
    args = message_dict['functionParameters'][0]['value']
    wait_for_sim()
    try:
        Msb.s.start_sim(args['x'], args['y'], args['n_agvs'])
    except KeyError as e:
        print("error using: ")
        print(args)



def callback_job(message_dict):
    args = message_dict['functionParameters'][0]['value']
    print(args)
    wait_for_sim()
    Msb.s.new_job(
        array([args['start_x'], args['start_y']]),
        array([args['goal_x'], args['goal_y']]),
        args['id']
    )


def callback_stop(message_dict):
    print("stop sim")
    wait_for_sim()
    Msb.s.stop()


def wait_for_sim():
    while not Msb.s:
        time.sleep(.1)


class Msb():
    s = False

    def __init__(self, s):
        Msb.s = s
        print("s: " + str(s))

        # Msb.mwc = MsbWsClient('ws://atm.virtualfortknox.de/msb', callback)
        # Msb.mwc = MsbWsClient('ws://ipa.virtualfortknox.de/msb', callback)
        Msb.mwc = MsbWsClient('ws://localhost:8085', callback)

        time.sleep(.1)

        # testing
        # s.new_job(
        #     array([1, 1]),
        #     array([20, 3])
        # )

        Msb.ePose = Event(
            eventId="Pose",
            name='Pose',
            description='The Pose of an AGV',
            dataFormat=ComplexDataFormat(
                properties=[
                    DataFormat("id", "Integer"),
                    DataFormat("x", "Float"),
                    DataFormat("y", "Float")
                ]
            )
        )
        Msb.eReached = Event(
            eventId="Reached",
            name="Reached",
            description="An AGV has reached a goal",
            dataFormat=DataFormat(doc_type="Integer")
        )
        Msb.eReachedStart = Event(
            eventId="ReachedStart",
            name="ReachedStart",
            description="An AGV has reached a start",
            dataFormat=ComplexDataFormat(
                properties=[
                    DataFormat("agvId", "Integer"),
                    DataFormat("jobId", "Integer")
                ]
            )
        )
        Msb.eAGVAssignment = Event(
            eventId="AGVAssignment",
            name="AGVAssignment",
            description="A Job was assigned to an AGV",
            dataFormat=ComplexDataFormat(
                properties=[
                    DataFormat("agvId", "Integer"),
                    DataFormat("jobId", "Integer")
                ]
            )
        )
        Msb.fStart = Function(
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
        Msb.fJob = Function(
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
        Msb.fStop = Function(
            functionId="Stop",
            name="Stop",
            description="Stop the Simulation",
            callback=callback_stop
        )
        Msb.application = Application(
            uuid="3785b920-3777-43ad-9199-b5362d9ef4b6",
                token="b5362d9ef4b6",
            name="AGV sim",
            description="Simulation of AGVs",
            events=[Msb.ePose, Msb.eReached, Msb.eAGVAssignment, Msb.eReachedStart],
            functions=[Msb.fStart, Msb.fJob, Msb.fStop]
        )

        Msb.mwc.register(Msb.application)
