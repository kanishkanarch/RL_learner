#!/usr/bin/env python3
import pdb

import rospy
from sensor_msgs.msg import Image
from airsim_ros_pkgs.msg import VelCmd
from nav_msgs.msg import Odometry

import random
import cv2
from cv_bridge import CvBridge
import numpy as np
import airsim

import torch
from torchvision import transforms as ttf

rospy.init_node("copy_node")

class reward_class:
    def __init__(self):
        self.random_poses = [
                #(0, -1, 0)
                (0, -1, 0, 0, 0, 0),
                (0, -1, 0, 0, 0, np.pi/2),
                (0, -1, 0, 0, 0, np.pi),
                (0, -1, 0, 0, 0, -np.pi/2),

                #(80.5, -1, 0)
                (80.5, -1, 0, 0, 0, 0),
                (80.5, -1, 0, 0, 0, np.pi/2),
                (80.5, -1, 0, 0, 0, np.pi),
                (80.5, -1, 0, 0, 0, -np.pi/2),

                #(128.5, -1, 0)
                (128.5, -1, 0, 0, 0, np.pi/2),
                (128.5, -1, 0, 0, 0, np.pi),
                (128.5, -1, 0, 0, 0, -np.pi/2),

                #(128.5, 126.5, 0)
                (128.5, 126.5, 0, 0, 0, np.pi),
                (128.5, 126.5, 0, 0, 0, -np.pi/2),

                #(0, 126.5, 0)
                (0, 126.5, 0, 0, 0, 0),
                (0, 126.5, 0, 0, 0, np.pi),
                (0, 126.5, 0, 0, 0, -np.pi/2),

                #(-127.5, 126.5, 0)
                (-127.5, 126.5, 0, 0, 0, 0),
                (-127.5, 126.5, 0, 0, 0, -np.pi/2),

                #(-127.5, -1, 0)
                (-127.5, -1, 0, 0, 0, 0),
                (-127.5, -1, 0, 0, 0, np.pi/2),
                (-127.5, -1, 0, 0, 0, -np.pi/2),

                #(-127.5, -128.5, 0)
                (-127.5, -128.5, 0, 0, 0, 0),
                (-127.5, -128.5, 0, 0, 0, np.pi/2),

                #(0, -128.5, 0)
                (0, -128.5, 0, 0, 0, 0),
                (0, -128.5, 0, 0, 0, np.pi/2),
                (0, -128.5, 0, 0, 0, np.pi),

                #(80.5, -128.5, 0)
                (80.5, -128.5, 0, 0, 0, 0),
                (80.5, -128.5, 0, 0, 0, np.pi/2),
                (80.5, -128.5, 0, 0, 0, np.pi),

                #(128.5, -128.5, 0)
                (128.5, -128.5, 0, 0, 0, np.pi/2),
                (128.5, -128.5, 0, 0, 0, np.pi),
        ]
        self.seg_sub = rospy.Subscriber("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, self.seg_cb)
        self.color_sub = rospy.Subscriber("/airsim_node/SimpleFlight/color/Scene", Image, self.color_cb)
        self.odom_sub = rospy.Subscriber("/airsim_node/SimpleFlight/odom_local_ned", Odometry, self.odom_cb)
        self.cmd_pub = rospy.Publisher("/airsim_node/SimpleFlight/vel_cmd_body_frame", VelCmd, queue_size = 10)

        self.seg_in = Image()
        self.color_in = Image()
        self.odom = Odometry()
        self.cmd_vel = VelCmd()
        self.cmd_vel.twist.linear.x = 1.0
        self.road_mask = None
        self.prev_img = Image()

        self.bridge = CvBridge()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.reset()
        prev_time = rospy.get_time()
        print("Resetting...")
        while rospy.get_time() < prev_time + 3:
            pass
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        prev_time = rospy.get_time()
        print("Arming...")
        while rospy.get_time() < prev_time + 5:
            pass
        self.client.takeoffAsync()
        prev_time = rospy.get_time()
        print("Taking off...")
        while rospy.get_time() < prev_time + 5:
            pass


        self.state_actions = np.array([-0.5, 0, 0.5])
        self.q_values = np.array([
                    [0, 0, 0], # Left
                    [0, 0, 0], # Straight
                    [0, 0, 0]  # Right
                ])
        self.epsilon = 0.05
        self.gamma = 0.1
        self.incremental_reward = 0
        self.step = 0
        self.reset_count = 0


    def odom_cb(self, msg):
        self.odom = msg

    def seg_cb(self, msg):
        self.seg_in = msg

    def color_cb(self, msg):
        self.color_in = msg

    def reset_env(self):
        self.client.reset()
        rndm_i = random.randint(0, len(self.random_poses)-1)
        pose = airsim.Pose(airsim.Vector3r(
            self.random_poses[rndm_i][0],
            self.random_poses[rndm_i][1],
            self.random_poses[rndm_i][2]
            ),
            airsim.to_quaternion(
            self.random_poses[rndm_i][3],
            self.random_poses[rndm_i][4],
            self.random_poses[rndm_i][5]
            )
        )
        self.client.simSetVehiclePose(pose, True, "SimpleFlight")
        self.client.enableApiControl(True)
        self.preferences = np.array([0, 0, 0])
        self.incremental_reward = 0
        self.step = 0
        self.reset_count += 1
        self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)

    def get_state(self, ros_image):
        # Convert the ROS image into CV2 image and mask everything except road
        image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding = "passthrough")
        img = image[:,:,0].copy()
        img[img != 246] = 0
        img[img == 246] = 255

        whites = np.count_nonzero(img, axis=0)
        if whites.max() == 0:
            return -1, -1
        indices = np.arange(img.shape[1])+1
        mean_section = (whites*indices).sum()/whites.sum()

        # Path center can be either in left, center, or right side of the image mask
        path_center = 1*(img > 0) + 1*(img > img.shape[1]/3) + 1*(img > 2*img.shape[1]/3)-1
        num_whites = whites.sum()
        
        return num_whites, path_center

    def calculate_reward(self):
        if self.client.simGetCollisionInfo().has_collided:
            return "reset"

        # Current observation
        current_num_whites, current_path_center = self.get_state(self.seg_in)
        if current_num_whites == -1:
            return "reset"
        
        # Previous observation
        prev_num_whites, prev_path_center = self.get_state(self.prev_img)

        prev_img = self.bridge.imgmsg_to_cv2(self.prev_img, desired_encoding = "passthrough")[:,:,0]
        # Cache current observation for next step
        self.prev_img = self.seg_in

        # Change in normalized area of road in image frame
        return 100*(current_num_whites - prev_num_whites)/(prev_img.shape[0]*prev_img.shape[1])

    def act(self):
        # Select epsilon-greedy action from self.state_actions using self.q_values
        self.prev_img = self.seg_in
        random_value = random.uniform(0, 1)
        explore = random_value < self.epsilon
        num_whites, path_center = self.get_state(self.seg_in)
        max_action_idx = np.argmax(self.q_values[path_center])
        random_action_idx = random.choice(list(range(0, max_action_idx)) + list(range(max_action_idx+1, 2)))
        action_idx = explore*random_action_idx + (1-explore)*max_action_idx

        # Repeat the selected action for 1 second
        prev_time = rospy.get_time()
        while rospy.get_time() - prev_time < 1:
            if self.step == 0:
                self.cmd_vel.twist.angular.z = self.state_actions[action_idx]
                self.cmd_pub.publish(self.cmd_vel)
            else:
                self.cmd_vel.twist.angular.z = self.state_actions[action_idx]
                self.cmd_pub.publish(self.cmd_vel)

        self.step += 1
        return path_center, action_idx

    def update_q(self, reward, path_center, action_idx):
        if reward == "reset":
            return
#        self.q_values[path_center][ # Continue from here

    def learn(self):
        self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)
        while not rospy.is_shutdown():
            path_center, action_idx = self.act()
            reward = self.calculate_reward()
            if reward == "reset":
                self.reset_env()
            self.update_q(reward, path_center, action_idx)

reward_class = reward_class()
reward_class.learn()
rospy.spin()
