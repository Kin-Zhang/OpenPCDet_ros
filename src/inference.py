#! /usr/bin/env python

"""
Created on Thu Aug  6 11:27:43 2020

@author: Javier del Egido Sierra and Carlos Gómez-Huélamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: LiDAR pointcloud
Outputs: Most relevant obstacles of the environment in the form of 3D bounding boxes

Note that each obstacle shows an unique ID in addition to its semantic information (person, car, ...), 
in order to make easier the decision making processes.

Executed via Python3.6 (python3.6 inference.py)
"""

# General use imports
import os
import time
import sys
import copy
import json
import argparse
import glob
from pathlib import Path

# ROS imports
import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Float64, Float32, Header, Bool
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import MarkerArray, Marker
from t4ac_msgs.msg import BEV_detection, BEV_detections_list

# Math and geometry imports
import math
import numpy as np
import torch
from pyquaternion import Quaternion

# OpenPCDet imports
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

# Global variables
calib_file = "/root/catkin_ws/src/OpenPCDet_ROS/calib_files/carla.txt"
cfg_root = "/root/OpenPCDet/tools/cfgs"

move_lidar_center = 20 
threshold = 0.7
rc = 0.0

image_shape = np.asarray([375, 1242])
inference_time_list = []

display_rviz = True
bev_camera = True

# Auxiliar functions 

def yaw2quaternion(yaw: float) -> Quaternion:
    """
    """
    return Quaternion(axis=[0,0,1], radians=yaw)

def rotz(t):
    """
    Rotation around 
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,  0],
                     [0,   0,  1]])

def cart2pol(x, y):
    """
    Transform cartesian to polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def get_annotations_indices(types, thresh, label_preds, scores):
    """
    """
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices 

def remove_low_score_nu(image_anno, thresh):
    """
    """
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.45, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.45, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.45, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.35, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.10, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    """
    """
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def anno_to_bev_detections(dt_box_lidar, scores, types, msg):
    """
    """
 
    xmax = rc * 10/130
    xmin = 0
    ymax = 1.5
    ymin = -2.0

    detected_3D_objects_marker_array = MarkerArray()

    bev_detections_list = BEV_detections_list()
    bev_detections_list.header.stamp = msg.header.stamp
    bev_detections_list.header.frame_id = msg.header.frame_id 

    point_cloud_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    bev_detections_list.front = point_cloud_range[3] - move_lidar_center
    bev_detections_list.back = point_cloud_range[0] - move_lidar_center
    bev_detections_list.left = point_cloud_range[1]
    bev_detections_list.right = point_cloud_range[4]

    if scores.size != 0:
        for i in range(scores.size):
            if scores[i] > threshold: 
                z = float(dt_box_lidar[i][2])
                l = float(dt_box_lidar[i][3])
                w = float(dt_box_lidar[i][4])
                yaw = float(dt_box_lidar[i][6])

                x_corners = [-l/2,-l/2,l/2, l/2]
                y_corners = [ w/2,-w/2,w/2,-w/2]
                z_corners = [0,0,0,0]

                if yaw > math.pi:
                    yaw -= math.pi

                if bev_camera: # BEV camera frame
                    x = -float(dt_box_lidar[i][1])
                    y = -(float(dt_box_lidar[i][0]) - move_lidar_center)
                    yaw_bev = yaw - math.pi/2
                else: # BEV LiDAR frame
                    x = float(dt_box_lidar[i][0]) - move_lidar_center
                    y = float(dt_box_lidar[i][1])
                    yaw_bev = yaw
  
                R = rotz(-yaw_bev)

                corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))[0:2]

                bev_detection = BEV_detection()
                bev_detection.type = str(int(types[i]))
                bev_detection.score = scores[i]

                bev_detection.x = x
                bev_detection.y = y
                bev_detection.tl_br = [0,0,0,0] #2D bbox top-left, bottom-right  xy coordinates
                                          # Upper left     Upper right      # Lower left     # Lower right
                bev_detection.x_corners = [corners_3d[0,0], corners_3d[0,1], corners_3d[0,2], corners_3d[0,3]]
                bev_detection.y_corners = [corners_3d[1,0], corners_3d[1,1], corners_3d[1,2], corners_3d[1,3]]
                bev_detection.l = l
                bev_detection.w = w
                bev_detection.o = -yaw_bev

                bev_detections_list.bev_detections_list.append(bev_detection)

                if display_rviz:
                    detected_3D_object_marker = Marker()
                    detected_3D_object_marker.header.stamp = msg.header.stamp
                    detected_3D_object_marker.header.frame_id = msg.header.frame_id
                    detected_3D_object_marker.type = Marker.CUBE
                    detected_3D_object_marker.id = i
                    detected_3D_object_marker.lifetime = rospy.Duration.from_sec(1)
                    detected_3D_object_marker.pose.position.x = float(dt_box_lidar[i][0]) - move_lidar_center
                    detected_3D_object_marker.pose.position.y = float(dt_box_lidar[i][1])
                    detected_3D_object_marker.pose.position.z = z
                    q = yaw2quaternion(yaw)
                    detected_3D_object_marker.pose.orientation.x = q[1] 
                    detected_3D_object_marker.pose.orientation.y = q[2]
                    detected_3D_object_marker.pose.orientation.z = q[3]
                    detected_3D_object_marker.pose.orientation.w = q[0]
                    detected_3D_object_marker.scale.x = l
                    detected_3D_object_marker.scale.y = w
                    detected_3D_object_marker.scale.z = 3
                    detected_3D_object_marker.color.r = 0
                    detected_3D_object_marker.color.g = 0
                    detected_3D_object_marker.color.b = 255
                    detected_3D_object_marker.color.a = 0.5
                    detected_3D_objects_marker_array.markers.append(detected_3D_object_marker)

    pub_detected_obstacles.publish(bev_detections_list)
    pub_rviz.publish(detected_3D_objects_marker_array)

    return 

def rslidar_callback(msg):
    """
    """

    frame = msg.header.seq
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)

    scores, dt_box_lidar, types, pred_dict = proc_1.run(np_p, calib, frame)
    anno_to_bev_detections(dt_box_lidar, scores, types, msg)

def rc_callback(msg):
    """
    """
    global rc
    rc = msg.data
                    
# Classes

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/muzi2045/Documents/project/OpenPCDet/data/kitti/velodyne/000001.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        print("Model path: ", self.model_path)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def get_calib(self, idx):
        return calibration_kitti.Calibration(calib_file)

    def get_template_prediction(self, num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def run(self, points, calib, frame):
        t_t = time.time()
        num_features = 4 # X,Y,Z,intensity       
        self.points = points.reshape([-1, num_features])

        frame = 0
        timestamps = np.empty((len(self.points),1))
        timestamps[:] = frame

        self.points = np.append(self.points, timestamps, axis=1)
        self.points[:,0] += move_lidar_center

        input_dict = {
            'points': self.points,
            'frame_id': frame,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        inference_time = time.time() - t
        inference_time_list.append(inference_time)
        mean_inference_time = sum(inference_time_list)/len(inference_time_list)

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        pred_boxes = np.copy(boxes_lidar)
        pred_dict = self.get_template_prediction(scores.shape[0])
        if scores.shape[0] == 0:
            return pred_dict

        pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )

        pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        pred_dict['bbox'] = pred_boxes_img
        pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = pred_boxes

        return scores, boxes_lidar, types, pred_dict
 
if __name__ == "__main__":
    config_path = os.path.join(cfg_root,"kitti_models/pointpillar.yaml")
    model_path  = os.path.join(cfg_root,"kitti_models/pointpillar_7728.pth")

    proc_1 = Processor_ROS(config_path, model_path)
    proc_1.initialize()
    calib = proc_1.get_calib(calib_file)
    calib.P2 = calib.P3
    print("Calib.P", calib.P2)
    print("Calib.P", calib.P3)
    print("Calib.R", calib.R0)
    print("Calib.T", calib.V2C)
    
    rospy.init_node('object_3d_detector_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/carla/ego_vehicle/lidar/lidar1/point_cloud",
                        "/kitti_player/hdl64e", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds",
                        "/livox/lidar",
                        "/SimOneSM_PointCloud_0"]

    cfg_from_yaml_file(config_path, cfg)
    
    sub_ = rospy.Subscriber(sub_lidar_topic[1], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    sub_ = rospy.Subscriber('/control/rc', Float64, rc_callback, queue_size=20)

    pub_detected_obstacles = rospy.Publisher('/perception/detection/bev_lidar_obstacles', BEV_detections_list, queue_size=5)
    pub_rviz = rospy.Publisher('/perception/detection/3d_lidar_obstacles_markers', MarkerArray, queue_size=5)
    
    print("[+] PCDet ros_node has started.")    
    rospy.spin()
