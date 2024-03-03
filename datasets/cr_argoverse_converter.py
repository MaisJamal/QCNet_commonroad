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
from commonroad.common.common_lanelet import LineMarking
#from commonroad.visualization.draw_params import DynamicObstacleParams

from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, create_collision_object
from commonroad.prediction.prediction import TrajectoryPrediction, SetBasedPrediction

import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request

import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from tqdm import tqdm

from utils import safe_list_index
from utils import side_to_directed_lineseg

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.lane_segment import LaneSegment,LaneType,LaneMarkType
    from av2.map.map_primitives import Polyline,Point
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object


def converter(scenario, planning_problem_set):
    argo_map =ArgoverseStaticMap(log_id="1",vector_drivable_areas={},vector_lane_segments={},
                                vector_pedestrian_crossings = {},raster_drivable_area_layer={}, raster_roi_layer={},raster_ground_height_layer={}) 
    argo_map.log_id = "0"
    #lane_seg = LaneSegment(id="2",is_intersection=False,lane_type)
    #'id', 'is_intersection', 'lane_type', 'right_lane_boundary', 'left_lane_boundary', 'right_mark_type', 'left_mark_type', 'predecessors', and 'successors'
    #lane_seg.id = "1"
    lanelets = scenario.lanelet_network.lanelets
    segments = []
    centerlines = {}
    print("len of lanelets ", len(scenario.lanelet_network.lanelets))
    print("len of polygons ", len(scenario.lanelet_network.intersections))
    for lane in lanelets:
        seg_id = int(lane.lanelet_id)
        seg_lane_type = LaneType.VEHICLE
        waypoints = []
        for point in lane.left_vertices:
            argo_pt = Point(x= point[0],y=point[1],z=0.0)
            waypoints.append(argo_pt)
        seg_left_bound = Polyline(waypoints=waypoints)
        seg_left_mark = get_lane_mark(lane.line_marking_left_vertices)

        waypoints = []
        for point in lane.right_vertices:
            argo_pt = Point(x= point[0],y=point[1],z=0.0)
            waypoints.append(argo_pt)
        seg_right_bound = Polyline(waypoints=waypoints)
        seg_right_mark = get_lane_mark(lane.line_marking_right_vertices)
        predecessors = lane.predecessor
        successors =  lane.successor
        lane_seg = LaneSegment(id=seg_id,is_intersection=False,lane_type=seg_lane_type,right_lane_boundary=seg_right_bound,left_lane_boundary=seg_left_bound,
                               right_mark_type= seg_right_mark,left_mark_type=seg_left_mark,predecessors=predecessors,successors =successors)
        waypoints = []
        for point in lane.center_vertices:
            argo_pt = Point(x= point[0],y=point[1],z=0.0)
            waypoints.append(argo_pt)
        centerline = Polyline(waypoints=waypoints)
        mapping = {seg_id : centerline}
        centerlines.update(mapping)

        segments.append(lane_seg)
        data: Dict[int, LaneSegment] = {seg_id:lane_seg}
        argo_map.vector_lane_segments.update(data)
        
    print("len of segments ", len(argo_map.vector_lane_segments))    

    ### convert agent data to argoverse dataframe (pandas)
    data = {
    "track_id": [],
    "timestep": [],
    "object_type":[],
    "object_category":[],
    "position_x":[],
    "position_y":[],
    "heading":[],
    "velocity_x":[],
    "velocity_y":[]
    }

    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    INIT_STATE = planning_problem.initial_state

    for dyn_obst in scenario.dynamic_obstacles:
        for i in range(50):
            if dyn_obst.state_at_time(i) == None:
                print("Error: Commonroad scenario doesn't cover 5 seconds.")
                break
            obs_x = dyn_obst.state_at_time(i).position[0]
            obs_y = dyn_obst.state_at_time(i).position[1] 
            obs_id = dyn_obst.obstacle_id
            obs_heading = dyn_obst.state_at_time(i).orientation
            obs_v_x = dyn_obst.state_at_time(i).velocity
            obs_v_y = 0.0
            data["track_id"].append(obs_id)
            data["timestep"].append(i)
            data["object_type"].append("vehicle")
            data["object_category"].append(1)
            data["position_x"].append(obs_x)
            data["position_y"].append(obs_y)
            data["heading"].append(obs_heading)
            data["velocity_x"].append(obs_v_x)
            data["velocity_y"].append(obs_v_y)

            ## append ego vehicle state
            data["track_id"].append("AV")
            data["timestep"].append(i)
            data["object_type"].append("vehicle")
            data["object_category"].append(1)
            data["position_x"].append(INIT_STATE.position[0])
            data["position_y"].append(INIT_STATE.position[1])
            data["heading"].append(INIT_STATE.orientation)
            data["velocity_x"].append(0)
            data["velocity_y"].append(0)

    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    """
    ['track_id']   = '8495' or 'AV'
    ['timestep']  = 0 , 1 ,.., 49
    ['object_type']  = vehicle
    ['object_category'] = 1
    ['position_x'] 
    ['position_y']
    ['heading']   pi
    ['velocity_x'] -11 or + 10
    ['velocity_y']
    """
    
    return argo_map, centerlines,df


def get_lane_mark(cr_mark):
    if cr_mark == LineMarking.DASHED:
        return LaneMarkType.DASHED_WHITE
    elif cr_mark == LineMarking.SOLID:
        return LaneMarkType.DASH_SOLID_WHITE
    elif cr_mark == LineMarking.BROAD_DASHED:
        return LaneMarkType.DOUBLE_DASH_WHITE
    elif cr_mark == LineMarking.BROAD_SOLID:
        return LaneMarkType.DOUBLE_SOLID_YELLOW
    elif cr_mark == LineMarking.UNKNOWN:
        return LaneMarkType.UNKNOWN
    elif cr_mark == LineMarking.NO_MARKING:
        return LaneMarkType.NONE
        