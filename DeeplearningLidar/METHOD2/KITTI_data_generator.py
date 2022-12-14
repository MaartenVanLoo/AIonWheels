import argparse
import glob
import os
import sys
import traceback
from pathlib import Path
import random

try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/CARLA_0.9.10/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.10",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
from datetime import date
from modules import generator_KITTI as gen
from GenerateLabels import Labels
from tqdm import tqdm

def main(args):
    map = args.map
    assert(map >= 0 and map < 8) #7 maps from Town01 to Town07 and 10
    i_map = list([1, 2, 3, 4, 5, 6, 7, 10])[map]
    start_record_full = time.time()

    fps_simu = 40.0
    time_stop = 2.0
    nbr_frame = 2000 #MAX = 10000
    nbr_walkers = 30
    nbr_vehicles = 30

    actor_list = []
    vehicles_list = []
    all_walkers_id = []
    data_date = date.today().strftime("%Y_%m_%d")
    
    spawn_points = [23,46,0,125,53,257,62,0]
    
    init_settings = None

    config = {
        'capture_freq':1 # Hz
    }
    world = None
    client = None
    try:
        #client = carla.Client('192.168.0.98', 2000)
        client = carla.Client(args.host, 2000)
        init_settings = carla.WorldSettings()


        client.set_timeout(100.0)
        print("Map Town0"+str(i_map))
        print(client.get_available_maps())
        if (i_map == 10):
            world = client.load_world("/Game/Carla/Maps/Town"+str(i_map)+"HD", reset_settings=False)
        else:
            world = client.load_world("/Game/Carla/Maps/Town0"+str(i_map), reset_settings=False)
        folder_output = "KITTI_Dataset_CARLA_v%s/%s/generated" %(client.get_client_version(), world.get_map().name)
        os.makedirs(folder_output) if not os.path.exists(folder_output) else [os.remove(f) for f in glob.glob(folder_output+"/*") if os.path.isfile(f)]
        client.start_recorder(os.path.dirname(os.path.realpath(__file__))+"/"+folder_output+"/recording.log")

        # Weather
        world.set_weather(carla.WeatherParameters.WetCloudyNoon)

        # Set Synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0/fps_simu
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        # Create KITTI vehicle
        blueprint_library = world.get_blueprint_library()
        bp_KITTI = blueprint_library.find('vehicle.tesla.model3')
        bp_KITTI.set_attribute('color', '228, 239, 241')
        bp_KITTI.set_attribute('role_name', 'KITTI')
        start_pose = world.get_map().get_spawn_points()[spawn_points[map]]
        KITTI = world.spawn_actor(bp_KITTI, start_pose)
        waypoint = world.get_map().get_waypoint(start_pose.location)
        actor_list.append(KITTI)
        print('Created %s' % KITTI)

        # Spawn vehicles and walkers
        gen.spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id)

        # Wait for KITTI to stop
        start = world.get_snapshot().timestamp.elapsed_seconds
        print("Waiting for KITTI to stop ...")
        while world.get_snapshot().timestamp.elapsed_seconds-start < time_stop: world.tick()
        print("KITTI stopped")

        # Set sensors transformation from KITTI
        lidar_transform     = carla.Transform(carla.Location(x=0, y=0, z=1.80), carla.Rotation(pitch=0, yaw=0, roll=0))
        cam0_transform = carla.Transform(carla.Location(x=0.30, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))       #left
        #cam1_transform = carla.Transform(carla.Location(x=0.30, y=0.50, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))    #right

        # Take a screenshot
        gen.screenshot(KITTI, world, actor_list, folder_output, carla.Transform(carla.Location(x=0.0, y=0, z=2.0), carla.Rotation(pitch=0, yaw=0, roll=0)))

        # Create our sensors
        gen.RGB.sensor_id_glob = 0
        gen.SS.sensor_id_glob = 10
        gen.Depth.sensor_id_glob = 20
        gen.HDL64E.sensor_id_glob = 100
        #VelodyneHDL64 = gen.HDL64E(KITTI, world, actor_list, folder_output, lidar_transform)
        lidar = gen.LIDAR(KITTI,world, actor_list, folder_output, lidar_transform,config)
        semantic_lidar = gen.SemanticLidar(KITTI,world, actor_list, folder_output, lidar_transform,config)
        cam0 = gen.RGB(KITTI, world, actor_list, folder_output, cam0_transform,config)
        #cam1 = gen.RGB(KITTI, world, actor_list, folder_output, cam1_transform,config)
        #cam0_ss = gen.SS(KITTI, world, actor_list, folder_output, cam0_transform,config)
        #cam1_ss = gen.SS(KITTI, world, actor_list, folder_output, cam1_transform,config)
        #cam0_depth = gen.Depth(KITTI, world, actor_list, folder_output, cam0_transform,config)
        #cam1_depth = gen.Depth(KITTI, world, actor_list, folder_output, cam1_transform,config)
        labels = Labels(folder_output)
        # Export LiDAR to cam0 transformation
        tf_lidar_cam0 = gen.transform_lidar_to_camera(lidar_transform, cam0_transform)
        with open(folder_output+"/lidar_to_cam0.txt", 'w') as posfile:
            posfile.write("#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
            posfile.write(str(tf_lidar_cam0[0][0])+" "+str(tf_lidar_cam0[0][1])+" "+str(tf_lidar_cam0[0][2])+" "+str(tf_lidar_cam0[0][3])+" ")
            posfile.write(str(tf_lidar_cam0[1][0])+" "+str(tf_lidar_cam0[1][1])+" "+str(tf_lidar_cam0[1][2])+" "+str(tf_lidar_cam0[1][3])+" ")
            posfile.write(str(tf_lidar_cam0[2][0])+" "+str(tf_lidar_cam0[2][1])+" "+str(tf_lidar_cam0[2][2])+" "+str(tf_lidar_cam0[2][3]))

        # Export LiDAR to cam1 transformation
        #tf_lidar_cam1 = gen.transform_lidar_to_camera(lidar_transform, cam1_transform)
        #with open(folder_output+"/lidar_to_cam1.txt", 'w') as posfile:
        #    posfile.write("#R(0,0) R(0,1) R(0,2) t(0) R(1,0) R(1,1) R(1,2) t(1) R(2,0) R(2,1) R(2,2) t(2)\n")
        #    posfile.write(str(tf_lidar_cam1[0][0])+" "+str(tf_lidar_cam1[0][1])+" "+str(tf_lidar_cam1[0][2])+" "+str(tf_lidar_cam1[0][3])+" ")
        #    posfile.write(str(tf_lidar_cam1[1][0])+" "+str(tf_lidar_cam1[1][1])+" "+str(tf_lidar_cam1[1][2])+" "+str(tf_lidar_cam1[1][3])+" ")
        #    posfile.write(str(tf_lidar_cam1[2][0])+" "+str(tf_lidar_cam1[2][1])+" "+str(tf_lidar_cam1[2][2])+" "+str(tf_lidar_cam1[2][3]))


        # Launch KITTI
        KITTI.set_autopilot(True)

        # Pass to the next simulator frame to spawn sensors and to retrieve first data
        world.tick()

        #VelodyneHDL64.init()
        gen.follow(KITTI.get_transform(), world)

        # All sensors produce first data at the same time (this ts)
        gen.Sensor.initial_ts = world.get_snapshot().timestamp.elapsed_seconds

        start_record = time.time()
        print("Start record : ")
        frame_current = 0
        prev_frame = 0
        #while for(frame_current < nbr_frame):
        total_ticks = int(nbr_frame / config.get('capture_freq',10) * fps_simu)
        for frame in tqdm(range(total_ticks)):
            #frame_current = VelodyneHDL64.save()
            frame_current = lidar.save(frame,fps_simu) #actual tick time = frame/fps_simu (in seconds)
            veh, ped = semantic_lidar.save(frame,fps_simu)
            cam0.save(no_id=True)
            #cam1.save()
            #cam0_ss.save(no_id=True)
            #cam1_ss.save()
            #cam0_depth.save(no_id=True)
            #cam1_depth.save()
            if prev_frame != frame_current:
                labels.genLabels(world,KITTI,lidar.sensor, veh, ped, prev_frame)
            prev_frame = frame_current
            gen.follow(KITTI.get_transform(), world)
            world.tick()    # Pass to the next simulator frame
            #time.sleep(0.05)

        #VelodyneHDL64.save_poses()
        client.stop_recorder()
        print("Stop record")

        #reset world settings
        world.apply_settings(init_settings)

        print('Destroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        vehicles_list.clear()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        all_actors = world.get_actors(all_walkers_id)
        for i in range(0, len(all_walkers_id), 2):
            all_actors[i].stop()
        print('Destroying %d walkers' % (len(all_walkers_id)//2))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
        all_walkers_id.clear()

        print('Destroying KITTI')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        actor_list.clear()

        print("Elapsed time : ", time.time()-start_record)
        print()

        world.wait_for_tick()
        time.sleep(2.0)
    except:
        if client:
            client.stop_recorder()
            print("Stop record")
        print(vehicles_list)
        print(all_walkers_id)
        print(actor_list)

        #reset world settings
        if world:
            print(world.get_actors().filter('vehicle.*'))
            world.apply_settings(init_settings)
            world.wait_for_tick()

        if vehicles_list:
            print('Destroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list.clear()

        # Stop walker controllers (list is [controller, actor, controller, actor ...])
        if all_walkers_id:
            all_actors = world.get_actors(all_walkers_id)
            for i in range(0, len(all_walkers_id), 2):
                all_actors[i].stop()
            print('Destroying %d walkers' % (len(all_walkers_id)//2))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_walkers_id])
            all_walkers_id.clear()

        if actor_list:
            print('Destroying KITTI')
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
            actor_list.clear()

        traceback.print_exc()
        print()
    finally:
        print("Elapsed total time : ", time.time()-start_record_full)
        if world and init_settings:
            world.apply_settings(init_settings)
        
        time.sleep(2.0)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP of carla host')
    parser.add_argument('--map', type=int, default=2,
                        help='The map to run')
    args = parser.parse_args()
    main(args)
