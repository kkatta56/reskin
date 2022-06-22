import sys
import os
from datetime import datetime
import platform

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import libraries.dobot_python.DobotDllType_Linux as dType 

#if platform.system() == 'Linux':
#    from .libraries.dobot_python import DobotDllType_Linux as dType
#elif platform.system() == 'Windows':
#    from .libraries.dobot_python import DobotDllType_Windows as dType

'''
    Add description here
'''

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

class Dobot:
    def __init__(self, port, baud_rate=115200, home_dobot=True):
        self.status = dType.DobotConnect.DobotConnect_NotFound
        self._initialize(port, baud_rate, home_dobot)
        self.mode = dType.PTPMode.PTPMOVLXYZMode

    def _initialize(self,port, baud_rate=115200, home_dobot=True, verbosity=1):
        # Load Dll and get the CDLL object
        self.api = dType.load()
        # Connect Dobot
        self.status = dType.ConnectDobot(self.api, port, baud_rate)[0]
        print("Connect status:",CON_STR[self.status])

        if (self.status == dType.DobotConnect.DobotConnect_NoError):

            # Clean Command Queued
            dType.SetQueuedCmdClear(self.api)
            dType.SetQueuedCmdStopExec(self.api)
            print('Cleared queue')

            # DO NOT CHANGE HOME PARAMS, DOBOT GLITCHES OUT OTHERWISE
            dType.SetHOMEParams(self.api, 220, 0, -6.5, 0, isQueued=1)
            dType.SetPTPJointParams(self.api,200,200,200,200,200,200,200,200, isQueued=1)
            dType.SetPTPCoordinateParams(self.api,200,200,200,200, isQueued=1)
            dType.SetPTPJumpParams(self.api, 10, 200, isQueued=1)
            lastIndex = dType.SetPTPCommonParams(self.api, 100, 100, isQueued=1)[0]
            print('Queued PTP params')
            # print(dType.GetHOMEParams(self.api))
            # DO NOT DO THIS RIGHT NOW. HOMING COMMAND IS SOME WEIRD SHIT.
            # The exact fix is available on one of the dobot forums where they 
            # tell you an exaCT POSITION TO USE 
            if home_dobot:
                while True:
                    check = input("Preparing to home dobot. Ensure no obstacles and press Y. Press N to skip.")
                    if check == "Y" or check == "y":
                        lastIndex = dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)[0]
                        print('Proceeding to home.')
                        break
                    elif check == 'N' or check == 'n':
                        print('Skipped homing.')
                        break
                    else: 
                        print('Invalid input.')

            # print('Queued homing command')

            dType.SetQueuedCmdStartExec(self.api)
            print('Starting Execution')
            # Wait for Executing Last Command 
            while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
                # print(lastIndex, dType.GetQueuedCmdCurrentIndex(self.api)[0])
                dType.dSleep(100)
            print('Initialization complete!')
            dType.SetQueuedCmdStopExec(self.api)
            dType.SetQueuedCmdClear(self.api)
        # return status, api

    def setPTPCoordinateParams(self, velocity):
        dType.SetPTPCoordinateParams(self.api,velocity,velocity,velocity,velocity, isQueued=0)

    def getPose(self):
        return dType.GetPose(self.api)
    
    def setOrigin(self, origin=None):
        
        if origin is not None:
            assert len(origin) == 3
            # origin[-1] += 10.
            origin_r = origin+[0.]
            origin_r[2] += 10.
            # print(origin_r)
            cmd_id = dType.SetPTPCmd(self.api, self.mode, *origin_r)
            print('Moving to 10 mm above specified origin', origin_r)
            dType.SetQueuedCmdStartExec(self.api)

            while cmd_id > dType.GetQueuedCmdCurrentIndex(self.api):
                dType.dSleep(100.)

            dType.SetQueuedCmdStopExec(self.api)
            dType.dSleep(100.)
            self.origin = dType.GetPose(self.api)
            # print(self.origin)
            self.origin[2] -= 10.
            print('Sensor reference successfully set.')
            print(self.origin)
            return self.origin

        else:
            while True:
                check = input("Move robot to bottom left screw on sensor (when viewed from the robot). Press Y when done. ")
                if check == "Y" or check == "y":
                    self.origin = dType.GetPose(self.api)
                    print('Sensor reference successfully set.')
                    print(self.origin)
                    return self.origin
                else: 
                    print('Invalid input.')

    def move(self,r,queue_cmd = False,delay=100.):
        
        assert len(r) == 3 or len(r) == 4
        if len(r) == 3:
            r += [0]
        # print(self.origin[:3])
        rel_r = np.array(self.origin[:4]) + np.array(r)
        # print(rel_r)

        if queue_cmd:
            dType.SetQueuedCmdStopExec(self.api)
        self.cmd_id = dType.SetPTPCmd(self.api, self.mode, *rel_r)  
        
        if not queue_cmd:
            dType.SetQueuedCmdStartExec(self.api)

            while self.cmd_id > dType.GetQueuedCmdCurrentIndex(self.api):
                dType.dSleep(delay)

            dType.SetQueuedCmdStopExec(self.api)
            dType.dSleep(delay)
        # print(dType.GetPose(self.api))
        
    def checkConnection(self):
        return (self.status == dType.DobotConnect.DobotConnect_NoError)
    
    def startQueueExec(self):
        dType.SetQueuedCmdStartExec(self.api)
    
    def stopQueueExec(self):
        dType.SetQueuedCmdStopExec(self.api)
    
    def checkQueueComplete(self):
        return self.cmd_id <= dType.GetQueuedCmdCurrentIndex(self.api)

    def startSnakePath(self, r):
        self.setOrigin(r)
        x = 0;
        y = 0;
        z = 0;
        xmove = 2;
        ymove = 2;
        zmove = 8;

        for j in range(10):
            for i in range(10):
                print([x,y])
                if not ((x < 4 or x > 12) and (y < 4 or y > 12)):
                    self.move([x, y, z])
                    self.move([x, y, z - zmove])
                    self.move([x, y, z])
                x += xmove
            xmove *= -1
            x += xmove
            y += ymove

    def upDown(self, r):
        self.setOrigin(r)
        n = input("How many indentations should the robot make?")
        for i in range(int(n)):
            self.move([8, 8, 0])
            self.move([8, 8, -8])
        self.move([8, 8, 0])

##################### ORIGINS #####################
# [177.79611206054688, -197.7755126953125, -87.25672912597656]
# [177.79611206054688, -95.56216430664062, -87.25672912597656]
