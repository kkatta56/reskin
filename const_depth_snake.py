import numpy as np
# import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm

from utils.dobot import *
from utils.sensor import *
from utils.force_sensor import ForceSensor, _ForceSensorSetting
import utils.libraries.dobot_python.DobotDllType_Linux as dType

# Could be modified later to change how we store stuff.
def save_data(robot_data, sensor_data, fs_data=None, save_aggr=False, directory='./expts', format='numpy'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Add assertion to ensure that robot data and sensor data has same number of rows 
    # Repeat robot data to match the number of sensor datapoints
    # 
    robot_data = np.array(robot_data)
    sensor_data = np.array(sensor_data)
    print(sensor_data.shape)
    print(directory)
    if fs_data is None:
        if not save_aggr:
            np.save(os.path.join(directory,'robot_data.npy'), robot_data)
            save_sensor_data(sensor_data, directory)
        else:
            robot_data_rep = np.repeat(robot_data, sensor_data.shape[1], axis=0)
            aggr_data = np.concatenate([robot_data_rep, sensor_data.reshape((-1,sensor_data.shape[-1]))], axis=1)
            if format == 'numpy' or format == 'np':
                # np.save(os.path.join(directory, 'aggr_data.npy'), aggr_data)
                with open(os.path.join(directory,'aggr_data.txt'), 'ab') as f:
                    np.savetxt(f, aggr_data) 

    else:
        fs_data = np.array(fs_data)
        print(fs_data.shape, sensor_data.shape)
        if not save_aggr:
            np.save(os.path.join(directory,'robot_data.npy'), robot_data)
            save_sensor_data(sensor_data, directory)
            np.save(os.path.join(directory,'fs_data.npy'), fs_data)
        else:
            robot_data_rep = np.repeat(robot_data, sensor_data.shape[1], axis=0)
            aggr_data = np.concatenate([robot_data_rep, 
                                        sensor_data.reshape((-1,sensor_data.shape[-1])),
                                        fs_data.reshape((-1,fs_data.shape[-1]))], axis=1)
            if format == 'numpy' or format == 'np':
                # np.save(os.path.join(directory, 'aggr_data.npy'), aggr_data)
                with open(os.path.join(directory,'aggr_data.txt'), 'ab') as f:
                    np.savetxt(f, aggr_data) 
    print('Saved successfully')
    # elif format == 'pandas':
        # pass

def plot_data():
    robot1 = np.load('./expts/const_depth_snake/latest-1/sensor_1/robot_data.npy')
    sensor1 = np.load('./expts/const_depth_snake/latest-1/sensor_1/sensor_data.npy')

    print(robot1.shape)
    print(sensor1.shape)

    sensor_means = np.mean(sensor1,axis=1)
    sensor_base = np.mean(sensor_means,axis=0)
    
    deviations = np.abs(sensor_means - sensor_base)

    heatmap_data = np.ones((5,5,15)) * np.mean(deviations, axis=0)
    print(robot1.shape, sensor1.shape)
    for i in range(robot1.shape[0]):
        x_ind = int((robot1[i,0]-2)/4)
        y_ind = int((robot1[i,1]-2)/4)
        heatmap_data[x_ind,y_ind] = deviations[i]

    print(heatmap_data.shape)

    for i in range(15):
        plt.subplot(5,3,i+1)
        plt.imshow(heatmap_data[...,i])
        plt.axis('off')
        # plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # plot_data()
    # exit()
    # dType.load()
    # SENSOR_PORTS = ["COM20"]
    # DOBOT_PORT = "COM25"

    # Dobot params
    DOBOT_PORT = ""

    robot = Dobot(DOBOT_PORT, 115200, True)

    # TODO: Sensor params
    SENSOR_PORTS = ["/dev/ttyACM0","/dev/ttyACM1","/dev/ttyACM3","/dev/ttyACM3","/dev/ttyACM4","/dev/ttyACM5"]
    use_sensor = [True, False, False, False, False, False]
    
    serSensor = []
    dropout = np.zeros((4,))

    # TODO: Force sensor params
    use_fs = False

    fs_settings = _ForceSensorSetting(device_name_prefix="Dev",
                       device_ids = [1],
                       sensor_names = ["FT34108"],
                       calibration_folder="./utils",
                       reverse_scaling = {1: ["Fz"], 2:["Fz"]},  # key: device_id, parameter. E.g.:if x & z dimension of sensor 1 and z dimension of sensor 2 has to be flipped use {1: ["Fx", "Fz"], 2: ["Fz"]}
                       remote_control=False, ask_filename= False, write_Fx=True,
                       write_Fy=True, write_Fz=True, write_Tx=False, write_Ty=False,
                       write_Tz=False, write_trigger1=True, write_trigger2=False,
                       zip_data=True, convert_to_forces=True,
                       priority='normal')
    
    if use_fs:
        force_sensor = ForceSensor(fs_settings)
    
    # TODO: Dobot path params
    depth_limit = 2.1 # (DON'T CHANGE) Specify depth limit in mm. Arbit for now. Can later take as user input and bound
    depth_incs = [1.,1.2] # Depth increments in mm
    x_res = 2.0 # Specify step size in x 
    y_res = 2.0 # Specify step size in y 

    x_lim = 16.0 # Specify relative distance to traverse in x
    y_lim = 16.0 # Specify relative distance to traverse in y

    # Data collection params
    num_batches = 1
    num_dps_init = 10 # DONT CHANGE
    num_dps = 5 # Number of datapoints collected at every indentation
    num_iters = 5 # How many times it goes over the sensor
    save_freq = 1 # DONT CHANGE


    # # Origins with non-fs 3mm indent
    # ######################### DONT CHANGE #########################
    # origins_list = [[180.43569946289062, -200.4913330078125, -97.4061767578125],
    #                 [182.93569946289062, -100.23883819580078, -95.4061767578125],
    #                 [186.28679809570312, 79.21101684570312, -95.2061767578125], # Reduced z depth by -.2 mm on 02/03/2021 - 11:21AM
    #                 [184.91615295410156, 180.7327423095703, -97.9061767578125]]
    # ########################### DONT CHANGE #######################
    
    # # Origins with fs 3mm indent
    # ######################### DONT CHANGE #########################
    # origins_list = [[180.07, -200.79, -96.61], # Reduce z depth by -.5 mm on 02/26/2021 - 01:41 PM
    #                 [183.43, -101.23, -94.81], # Reduce z depth by -.4 mm on 02/23/2021 - 02:23 PM
    #                 [186.78, 78.71, -94.91], # Reduced z depth by -.2 mm on 02/03/2021 - 11:21AM
    #                 [184.71, 180.53, -98.31], # Reduicing x,y depth by -.2 mm on 02/16/2021 - 01:21AM - furth z-=.2 on 02/25
    #                 [70.57, 209.25, -97.51],
    #                 [65.47, -251.41, -96.71]]
    # ######################### DONT CHANGE #########################

    # Origins with fs 3mm indent
    ######################### DONT CHANGE #########################
    # origins_list = [[181.07, -200.29, -94.31], # Reduce z depth by -.5 mm on 02/26/2021 - 01:41 PM
    #                 [184.43, -100.43, -92.31], # Reduce z depth by -.4 mm on 02/23/2021 - 02:23 PM
    #                 [187.38, 80.41, -91.71], # Reduced z depth by -.2 mm on 02/03/2021 - 11:21AM
    #                 [186.21, 169.43, -93.71], # Reduicing x,y depth by -.2 mm on 02/16/2021 - 01:21AM - furth z-=.2 on 02/25
    #                 [71.57, 209.95, -92.71],
    #                 [65.97, -250.41, -93.81]]
    
    # TODO: Calibrate
    origins_list = [[177.51, -197.39, -77.19], # Moved up by 0.2 mm, 03/26. 18:42
                    [177.01, -95.93, -75.79],
                    [178.51, 77.44, -75.59],
                    [182.01, 166.64, -77.29],
                    [69.01, 202.84, -77.29], # pushed away y by 0.6 mm. 04/27, 17:15
                    [64.81, -245.34, -77.29] # pulled down by .2, night of 04/26, x-=.6 on 04/27, 17:15
    ]

    # origins_list = [[178.01, -197.39, -77.39], # Moved up by 0.2 mm, 03/26. 18:42
    #                 [178.01, -95.93, -76.09],
    #                 [180.01, 78.44, -75.59],
    #                 [181.01, 168.64, -77.29],
    #                 [69.01, 203.24, -77.09],
    #                 [64.91, -245.84, -77.09]
    # ]
    ######################### DONT CHANGE #########################
    z_eqs = np.array([  [],
                        [],
                        [],
                        [0.01962,0.03378]
    ])
    reverse_path = False
    # reverse_origins = [[196.07,-184.79, -97.01],
    #                    [199.43, -85.03, -95.21], # Reduce z depth by -.4 mm on 02/23/2021 - 02:23 PM
    #                    [202.78, 94.71, -95.31], # Reduced z depth by -.2 mm on 02/03/2021 - 11:21AM
    #                    [200.71, 196.53, -98.31], # Reduicing x,y depth by -.2 mm on 02/16/2021 - 01:21AM - furth z-=.2 on 02/25
    #                    [86.57, 225.25, -97.51],
    #                    [81.47, -235.41, -96.71]]

    intermed_pos = [216.28, 0.0, -45.71]

    data_path = os.path.join('./expts','const_depth_snake/')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expt_dir = os.path.join(data_path,dt_string)
    
    # Initialize sensors
    for i,port in enumerate(SENSOR_PORTS):
        if use_sensor[i]:
            serSensor += [Sensor(port, 115200, burst_mode=True)]
        else:
            serSensor += [None]
    
    for batch in tqdm(range(num_batches)):
        for sensor_id, sensor in enumerate(serSensor):
            if use_fs:
                force_sensor.start_recording()
            if not use_sensor[sensor_id]:
                # print(sensor_id, use_sensor[sensor_id])
                continue
            while True:
                if robot.checkConnection:
                    if not reverse_path:
                        robot.setOrigin(origins_list[sensor_id])
                    else:
                        robot.setOrigin(reverse_origins[sensor_id])
                    # robot.setOrigin()
                
                    x = np.arange(0.0,x_lim+x_res,x_res)
                    y = np.arange(0.0,y_lim+y_res,y_res)
                    # print(x)
                    z = -depth_limit
                    z_high = 10.0
                    
                    robot_path = []
                    sensor_data = []
                    fs_data = []
                    
                    
                    for _it in range(num_iters):
                        robot.move([0.0,0.0, z_high], queue_cmd=False)
                        data, _ = sensor.collect_data(num_dps)
                        robot_path += [[0.0,0.0, z_high]]
                        sensor_data += [data]
                        
                        if use_fs:
                            fs_data += force_sensor.get_data(num_dps)

                        if len(data) < num_dps:
                            use_sensor[sensor_id] = False
                            break
                        for d in depth_incs:
                            for i in range(x.shape[0]):
                                dirn = (-1.)**i 
                                for j in range(y.shape[0]):
                                    # if (x[i] == 0 or x[i] == x_lim) and (y[j] == 0. or y[j] == y_lim):
                                    #     continue
                                    if (x[i] < 4 or x[i] > x_lim-4) and (y[j] < 4 or y[j] > y_lim-4):
                                        continue
                                    # print(z_eqs[sensor_id]@np.asarray([x[i],y[j]]))
                                    z = -(depth_limit + d)# + z_eqs[sensor_id]@np.asarray([x[i],y[j]]))
                                    y_curr = -(dirn*y[j]+ (0.5-dirn/2)*y[-1])
                                    
                                    if not reverse_path:
                                        robot.move([x[i], -y_curr, z_high], queue_cmd=False)
                                        if use_sensor[sensor_id]:
                                            zero_data, _ = sensor.collect_data(num_dps)
                                            robot_path += [[x[i],-y_curr, z_high]] 
                                            sensor_data += [data]
                                        if use_fs:
                                            fs_data += force_sensor.get_data(num_dps)
                                        robot.move([x[i], -y_curr, z], queue_cmd=False)
                                    else:
                                        robot.move([-x[i], y_curr, z_high], queue_cmd=True)
                                        robot.move([-x[i], y_curr, z], queue_cmd=False)
                                    # print([x[i], dirn*y[j]+ (0.5-dirn/2)*y[-1], z])

                                    if use_sensor[sensor_id]:
                                        data, _ = sensor.collect_data(num_dps)
                                        if len(data) < num_dps:
                                            use_sensor[sensor_id] = False
                                            break
                                        # robot_path += [[x[i],y[j], z]] # This is wrong and needs to be fixed with next line
                                        robot_path += [[x[i],-y_curr, z]] 
                                        sensor_data += [data]

                                        if use_fs:
                                            fs_data += force_sensor.get_data(num_dps)

                            if not reverse_path:
                                robot.move([x[-1], dirn*y[-1]+ (0.5-dirn/2)*y[-1], z_high*3])
                                robot.move([0.0,0.0, z_high*3], queue_cmd=False)
                                time.sleep(1.)
                                data, _ = sensor.collect_data(num_dps)
                                robot_path += [[0.0,0.0, z_high*3]]
                                sensor_data += [data]

                                if use_fs:
                                    fs_data += force_sensor.get_data(num_dps)
                            else:
                                robot.move([-x[-1], -(dirn*y[-1]+ (0.5-dirn/2)*y[-1]), z_high*3])
                            if not use_sensor[sensor_id]:
                                break
                        if not use_sensor[sensor_id]:
                            break
                        if (_it+1) % save_freq == 0 or (_it+1) == num_iters:
                            # print(robot_path)
                            if not os.path.isdir(expt_dir):
                                os.mkdir(expt_dir)
                            if not use_fs:
                                save_data(robot_path, 
                                    sensor_data, 
                                    save_aggr=True,
                                    directory=os.path.join(expt_dir,'sensor_'+str(sensor_id+1)))
                            else:
                                save_data(robot_path,
                                    sensor_data,
                                    fs_data=fs_data,
                                    save_aggr=True,
                                    directory=os.path.join(expt_dir,'sensor_'+str(sensor_id+1)))

                            sensor.flush_data()
                            sensor_data = []
                            robot_path = []
                            fs_data = []
                    break
                else:
                    input('Robot connection failed. Press r to retry.')
            if use_fs:
                force_sensor.pause_recording()
            time.sleep(1.)
            if (sensor_id < 5 and np.sum(np.asarray(use_sensor[sensor_id:-1])) <= 1) or sensor_id == 5:
                robot.setOrigin(intermed_pos)
            # if sensor_id == 4:
            #     robot.setOrigin(intermed_pos)
    if use_fs:
        force_sensor.quit()