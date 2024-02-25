# import functions to read xml file and visualize commonroad objects
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import math
import copy
import os

################ collision checker ##############################
import numpy as np
import commonroad_dc.pycrcc as pycrcc
from commonroad_dc.boundary import boundary
from time import time
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from vehiclemodels import parameters_vehicle3
from commonroad.geometry.shape import Rectangle

#from commonroad.visualization.draw_params import DynamicObstacleParams

from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object
from commonroad.prediction.prediction import TrajectoryPrediction, SetBasedPrediction


SHIFT_IN_FFSTREAM_Y = 48.25
SCENARIO_ROTATION_DEGREES = 0

def get_trajectory_list(loaded, obb_hull_preprocess=True):
    traj_matrix=loaded["trajectories"].reshape(1000,-1)
    start_time=np.asfortranarray(loaded["start_time_step"])
    half_car_length, half_car_width=loaded["car_shape"]/2

    # create a trajectory batch
    trajectories_batch=pycrcc.OBBTrajectoryBatch(traj_matrix,start_time, half_car_length, half_car_width)
    # preprocess the trajectory batch for continuous collision detection using OBB Hull method (see also Tutorial 03)
    if obb_hull_preprocess:
        trajectories_batch.preprocess_()

    return trajectories_batch.to_tvobstacle()


def get_scenario_dynamic_obstacles_list(scenario):
    dyn_obstacles_list=list()
    for dyn_obst in scenario.dynamic_obstacles:
        if isinstance(dyn_obst.prediction,TrajectoryPrediction):
            co=create_collision_object(dyn_obst.prediction)
            # preprocess using obb hull for continuous collision detection
            co, err=trajectory_queries.trajectory_preprocess_obb_sum(co)
            dyn_obstacles_list.append(co)
        else:
            if isinstance(dyn_obst.prediction,SetBasedPrediction):
                co=create_collision_object(dyn_obst.prediction)
                dyn_obstacles_list.append(co)
            else:
                raise Exception('Unknown dynamic obstacle prediction type: ' + str(type(dyn_obst.prediction)))
    return dyn_obstacles_list

def GetRotationAndLaneWidth(scenario):
    # get lane rotation angle (theta)
    delta_x = scenario.lanelet_network.lanelets[0].left_vertices[-1][0] - scenario.lanelet_network.lanelets[0].left_vertices[0][0]
    delta_y = scenario.lanelet_network.lanelets[0].left_vertices[-1][1] - scenario.lanelet_network.lanelets[0].left_vertices[0][1]
    theta_rad = math.atan(delta_y/delta_x)
    if delta_x>0 and delta_y >0:
        theta_rad += + 3.14
    theta = math.degrees(theta_rad) 
    # get lane width
    center_vertices = scenario.lanelet_network.lanelets[0].center_vertices
    print(center_vertices[0])
    print(center_vertices[-1])
    print(len(center_vertices))
    print(len(scenario.lanelet_network.lanelets))
    print(scenario.lanelet_network.lanelets[0].left_vertices[0][1])
    print(scenario.lanelet_network.lanelets[0].right_vertices[0][1])
    print("raw width ",(scenario.lanelet_network.lanelets[0].left_vertices[0][1]-scenario.lanelet_network.lanelets[0].right_vertices[0][1]))
    lane_width = abs( (scenario.lanelet_network.lanelets[0].left_vertices[0][1]-scenario.lanelet_network.lanelets[0].right_vertices[0][1])* math.cos(theta_rad))
    if lane_width < 2 :
        lane_width = 3.5
    return theta,lane_width


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
################################################################

def extract_data(file_path):
    global SHIFT_IN_FFSTREAM_Y
    global SCENARIO_ROTATION_DEGREES
    # read in the scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # get ego initial state in the scenario
    INIT_STATE = planning_problem.initial_state

    # usefull parameters
    theta, lane_width = GetRotationAndLaneWidth(scenario)  #theta in degrees
    
    SCENARIO_ROTATION_DEGREES = theta


    print("theta in degrees is ", theta)
    print("lane width is ", lane_width)
    angle = -1*math.radians(SCENARIO_ROTATION_DEGREES)
    # get dynamic obstacles in the scenario
    total_time_steps = scenario.dynamic_obstacles[0].prediction.final_time_step
    print("total_time_steps: ",total_time_steps)
    future_trajectories_obstacles = []
    init_obstacles = []
    for dyn_obst in scenario.dynamic_obstacles:
        print("dyn",dyn_obst)
        #print(dyn_obst.obstacle_shape)
        obs_pos = dyn_obst.initial_state.position
        obs_pos[0],obs_pos[1] = rotate((INIT_STATE.position[0],INIT_STATE.position[1]),(obs_pos[0],obs_pos[1]),angle)
        obs_pos[0] = obs_pos[0] - INIT_STATE.position[0]
        if dyn_obst.obstacle_type == ObstacleType.TRUCK:
            obs_pos[0] -= (dyn_obst.obstacle_shape.length - 5.5)
        obs_pos[1] = obs_pos[1] + SHIFT_IN_FFSTREAM_Y -INIT_STATE.position[1]
        obs_orien = dyn_obst.initial_state.orientation - theta
        obs_vel = dyn_obst.initial_state.velocity
        #obs_acc = dyn_obst.initial_state.acceleration
        #obstacles.append((obs_pos[0],obs_pos[1],obs_orien,obs_vel))
        init_obstacles.append(((obs_pos[0], obs_pos[1]), (5.5, 2.5), (obs_vel, 0)))
        traj = []
        for i in range(1,total_time_steps):
            if dyn_obst.state_at_time(i) == None:
                total_time_steps = i
                break
            p_x = dyn_obst.state_at_time(i).position[0]
            p_y = dyn_obst.state_at_time(i).position[1] 
            
            p_x , p_y = rotate((INIT_STATE.position[0],INIT_STATE.position[1]),(p_x,p_y),angle)
           
            p_x = p_x - INIT_STATE.position[0]
            #if dyn_obst.obstacle_type == ObstacleType.TRUCK:
            #    p_x -= (dyn_obst.obstacle_shape.length - 5.5)
            p_y = p_y + SHIFT_IN_FFSTREAM_Y -INIT_STATE.position[1]
            
            p_v = dyn_obst.state_at_time(i).velocity
            print("x y v ",dyn_obst.state_at_time(i).position[0],dyn_obst.state_at_time(i).position[1],dyn_obst.state_at_time(i).velocity)
            traj.append((p_x,p_y,p_v))
        # fix error in obs velocity
        """
        for i in range(len(traj)):
            if traj[i][2] == 0 and i > 0:
                as_list = list(traj[i])
                delta_vx = (traj[i][0]-traj[i-1][0])/0.1
                delta_vy = (traj[i][1]-traj[i-1][1])/0.1
                as_list[2] = math.sqrt(delta_vx*delta_vx + delta_vy*delta_vy)
                as_tuple = tuple(as_list)
                traj[i] = as_tuple
        """
        future_trajectories_obstacles.append(traj)
        
    #print(future_trajectories_obstacles) ##  [[ (obs1_x1,obs1_y1,obs1_v1),(obs1_x2,obs1_y2,obs1_v2), ... ],[(obs2_x1,obs2_y1,obs2_v1),(obs2_x2,obs2_y2,obs2_v2) , ...]]
    #####################################
    ego_orientation_ff = INIT_STATE.orientation - theta
    ego_x_ff = 0 #INIT_STATE.position[0]
    ego_y_ff = SHIFT_IN_FFSTREAM_Y#INIT_STATE.position[1] + SHIFT_IN_FFSTREAM_Y
    
    # plot the planning problem and the scenario for the fifth time step
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    scenario.draw(rnd)#(rnd, draw_params={'time_begin': 5})
    planning_problem_set.draw(rnd)
    rnd.render()
    plt.show()

    return ego_x_ff,ego_y_ff,ego_orientation_ff,INIT_STATE.velocity , init_obstacles , theta, lane_width, future_trajectories_obstacles,total_time_steps

def draw_traj_with_scenario(trajectory,file_path):

    global SHIFT_IN_FFSTREAM_Y
    global SCENARIO_ROTATION_DEGREES

    # time step in CommonRoad is 0.1 second
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    initial_state = planning_problem.initial_state
    initial_state.time_step = 0
    state_list = []

    some_state = initial_state
    N = len(trajectory.x)
    print("len of traj is ",N)
    i = 0
    while i < N*2:
        # compute new position
        # add new state to state_list
        some_state=copy.deepcopy(initial_state)
        p_x = trajectory.x[int(i/2)]+ initial_state.position[0]
        p_y = trajectory.y[int(i/2)] + initial_state.position[1]- SHIFT_IN_FFSTREAM_Y
        angle = math.radians(SCENARIO_ROTATION_DEGREES)
        p_x , p_y = rotate((initial_state.position[0],initial_state.position[1]),(p_x,p_y),angle)
        p_yaw = trajectory.yaw[int(i/2)]
        some_state.position[0] = p_x 
        some_state.position[1] = p_y  
        some_state.orientation = p_yaw + angle # should it be in degrees??!! #TOCHECK
        some_state.time_step = i
        state_list.append(some_state)
        i+=2
    
    #for s in state_list:
    #    print("*****",s)
    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
    # create the planned trajectory starting at time step 1
    ego_vehicle_trajectory = Trajectory(initial_time_step=2, state_list=state_list[1:])
    # create the prediction using the planned trajectory and the shape of the ego vehicle
    ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,shape=ego_vehicle_shape)
    # the ego vehicle can be visualized by converting it into a DynamicObstacle
    ego_vehicle_type = ObstacleType.CAR
    ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                              obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                              prediction=ego_vehicle_prediction)
    #ego_params = DynamicObstacleParams()
    #ego_params.vehicle_shape.occupancy.shape.facecolor = "g"
    road_boundary_obstacle, road_boundary_sg_rectangles=boundary.create_road_boundary_obstacle(scenario)

    # plot the planning problem and the scenario for the fifth time step
    max_time_step = min(N*2 , 32)
    #draw_params = {'dynamic_obstacle': {'shape': {'rectangle': {'facecolor': 'g'}}}}
    #gif_path = os.path.join(os.getcwd(), os.path.dirname(__file__), 'gifs/tut2.gif')
    
    #rnd = MPRenderer()
    #rnd.create_video([scenario, ego_vehicle], gif_path, 2, 20, draw_params=draw_params, dt=1)
    for i in range(0, max_time_step ,2):
        plt.figure(figsize=(25, 10))
        rnd = MPRenderer()
        scenario.draw(rnd, draw_params={'time_begin': i})
        road_boundary_sg_rectangles.draw(rnd)
        ego_vehicle.draw(rnd, draw_params={'facecolor': 'green','time_begin': i})
        planning_problem_set.draw(rnd)
        rnd.render()
        plt.show()
    """
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    scenario.draw(rnd, draw_params={'time_begin': 5})
    ego_vehicle.draw(rnd, draw_params={'facecolor': 'green'})
    planning_problem_set.draw(rnd)
    rnd.render()
    plt.show()
    """

def all_functions(file_path):
    # generate path of the file to be opened
    # read in the scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    INIT_STATE = planning_problem.initial_state
    print ("initial state: ", INIT_STATE)
    for state in planning_problem.goal.state_list:
        print("goal state: ",state)

    for state in scenario.obstacle_states_at_time_step(0):
        print ("scenario obstacles: ",state)

    dyn_obstacles_list=list()
    for dyn_obst in scenario.dynamic_obstacles:
        print("dyn_obs: " ,dyn_obst)
        obs_pos = dyn_obst.initial_state.position
        obs_orien = dyn_obst.initial_state.orientation
        obs_vel = dyn_obst.initial_state.velocity
        #obs_acc = dyn_obst.initial_state.acceleration
        #print("dyn obs * pos,orient,vel,acc:", obs_pos ,obs_orien,obs_vel,obs_acc)
    #################################################################
    initial_state = planning_problem.initial_state
    initial_state.time_step = 0
    state_list = []
    
    orientation = initial_state.orientation
    some_state = initial_state
    for i in range(4):
        # compute new position
        # add new state to state_list
        some_state=copy.deepcopy(initial_state)
        some_state.position[0] = i
        some_state.time_step = i
        state_list.append(some_state)
    
    #for s in state_list:
    #    print("*****",s)
    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
    # create the planned trajectory starting at time step 1
    ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])
    # create the prediction using the planned trajectory and the shape of the ego vehicle
    ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,shape=ego_vehicle_shape)

    # the ego vehicle can be visualized by converting it into a DynamicObstacle
    ego_vehicle_type = ObstacleType.CAR
    ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                              obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                              prediction=ego_vehicle_prediction)
    #ego_params = DynamicObstacleParams()
    #ego_params.vehicle_shape.occupancy.shape.facecolor = "g"
    ################### collision checker #########################
    data=np.load("/home/mais/testCommonRoad/commonroad_io/commonroad/tutorials/collision_checker/USA_US101-3_3_T-1_waypoints.npz")
    print("data: ",data)
    loaded_data = {"car_shape": data["car_shape"], "trajectories": data["trajectories"], "start_time_step": data["start_time_step"]}
    data.close()

    trajectories = get_trajectory_list(loaded_data)
    waypoints=loaded_data["trajectories"]
    #print(" waypoints : ", waypoints)
    #print(" start_time_step : ", loaded_data["start_time_step"])
    car_shape=loaded_data["car_shape"]
    car_half_length, car_half_width = (car_shape/2)

    # create the road boundary with default method uses oriented rectangles
    road_boundary_obstacle, road_boundary_sg_rectangles=boundary.create_road_boundary_obstacle(scenario)
    road_polygons = boundary.create_road_polygons(scenario, method='whole_polygon', triangulate=False)

    # Draw an exemplary part of trajectory batch (here: 50 trajectories) and the road boundary
    n_traj_draw=1
    offset=350

    rnd = MPRenderer(figsize=(25, 10))
    scenario.draw(rnd)
    road_boundary_sg_rectangles.draw(rnd)
    ego_vehicle.draw(rnd, draw_params={'facecolor': 'green'})
    for tvo in trajectories[offset:offset+n_traj_draw]:
        tvo.draw(rnd, draw_params={'facecolor': 'green'})
    rnd.render()


    # check computation time over ten runs
    num_trials=10
    cur_time_1=time()
    for i in range(num_trials):
        # check if the trajectory collides with the road boundary
        ret=trajectory_queries.trajectories_collision_static_obstacles(trajectories, road_boundary_sg_rectangles, method='grid', num_cells=32, auto_orientation=True, optimize_triangles=True)
    cur_time_2 = time()

    print("%s out of %s checked trajectories do not collide with the road boundary." % (ret.count(-1), len(trajectories)))
    print("Time for %s trajectory checks: " % (len(trajectories)) + str((cur_time_2-cur_time_1)/num_trials)+ " sec.")


    dynamic_obstacles=get_scenario_dynamic_obstacles_list(scenario)
    # check computation time over ten runs
    num_trials=10

    cur_time_1=time()
    for i in range(num_trials):
        res_dynamic = trajectory_queries.trajectories_collision_dynamic_obstacles(trajectories, dynamic_obstacles, method='box2d')
    cur_time_2 = time()

    print("%s out of %s trajectories do not collide with the other vehicles" % (res_dynamic.count(-1), len(trajectories)))
    print("Time for %s trajectory checks: " % (len(trajectories)) + str((cur_time_2-cur_time_1)/num_trials)+ " sec.")
    ############################################################


    # plot the planning problem and the scenario for the fifth time step
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    scenario.draw(rnd)#(rnd, draw_params={'time_begin': 5})
    planning_problem_set.draw(rnd)
    rnd.render()
    plt.show()

    return INIT_STATE.position[0],INIT_STATE.position[1],INIT_STATE.orientation,INIT_STATE.velocity



def  extract_front_obstacle(obstacles,q_ego):
    ### obstacles is a list of tuples (x pos, y pos, linear velocity)
    ### q_ego is the configuration of the ego vehicle , a tuple (x pos, y pos, linear velocity)
    front_obs_idx = 0
    found_front_obs = False
    for i in range(len(obstacles)):
        if abs(obstacles[i][0][1] - q_ego[1]) < 1 :  # compare Y position
            if obstacles[i][0][0] > q_ego[0] :        # compare X position
                if not found_front_obs:
                    found_front_obs = True
                    front_obs_idx = i
                else:
                    if obstacles[i][0][0] < obstacles[front_obs_idx][0][0]: # found a closed front obstacle
                        front_obs_idx = i
    return front_obs_idx



def cr_scene_to_qcnet (scene_path):
    init_x, init_y,init_heading,init_speed,init_scene_obstacles,scen_rotation, lane_width,future_trajectories_obstacles,total_time_steps = extract_data(scene_path)
    data = init_x
    return data