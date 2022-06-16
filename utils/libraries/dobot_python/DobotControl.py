import threading
import platform 

if platform.system() == "Windows":
    pass 
elif platform.system() == "Linux":
    import DobotDllType_Linux as dType

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#Load Dll and get the CDLL object
api = dType.load()

#Connect Dobot
print(dType.SearchDobot(api))
state = dType.ConnectDobot(api, '/dev/ttyUSB0', 115200)[0]
print("Connect status:",CON_STR[state])

print(dType.DobotConnect.DobotConnect_NoError)
if (state == dType.DobotConnect.DobotConnect_NoError):
    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    dType.SetPTPJointParams(api,200,200,200,200,200,200,200,200)
    dType.SetPTPCoordinateParams(api,200,200,200,200)
    dType.SetPTPJumpParams(api, 10, 200)
    dType.SetPTPCommonParams(api, 100, 100)
    moveX=0;moveY=0;moveZ=10;moveFlag=-1
    pos = dType.GetPose(api)
    print(pos)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    rHead = pos[3]


    moveFlag *= -1
    for i in range(5):
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x+moveX, y+moveY, z+moveZ, rHead, isQueued=1)
        moveX += 10 * moveFlag
        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x+moveX, y+moveY, z+moveZ, rHead, isQueued=1)
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x+moveX, y+moveY, z, rHead, isQueued=1)[0]
        print(dType.GetPose(api))
    dType.dSleep(1000)
    dType.SetQueuedCmdStartExec(api)
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        # print(lastIndex, dType.GetQueuedCmdCurrentIndex(api)[0])
        dType.dSleep(100)

    dType.DisconnectDobot(api)
    exit()
    #设置运动参数
    #Async Motion Params Setting
    dType.SetHOMEParams(api, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)

    #回零
    #Async Home
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    #设置ptpcmd内容并将命令发送给dobot
    #Async PTP Motion
    for i in range(0, 5):
        if i % 2 == 0:
            offset = 50
        else:
            offset = -50
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200 + offset, offset, offset, offset, isQueued = 1)[0]

    #开始执行指令队列
    #Start to Execute Command Queue
    dType.SetQueuedCmdStartExec(api)

    #如果还未完成指令队列则等待
    #Wait for Executing Last Command 
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        # print(lastIndex, dType.GetQueuedCmdCurrentIndex(api)[0])
        dType.dSleep(100)

    #停止执行指令
    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)

#断开连接
#Disconnect Dobot
dType.DisconnectDobot(api)
