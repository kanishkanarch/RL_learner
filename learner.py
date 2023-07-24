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
        self.seg_sub = rospy.Subscriber("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, self.seg_cb)
        self.color_sub = rospy.Subscriber("/airsim_node/SimpleFlight/color/Scene", Image, self.color_cb)
        self.odom_sub = rospy.Subscriber("/airsim_node/SimpleFlight/odom_local_ned", Odometry, self.odom_cb)
        self.img_pub = rospy.Publisher("/easytopic", Image, queue_size = 5)
        self.cmd_pub = rospy.Publisher("/airsim_node/SimpleFlight/vel_cmd_body_frame", VelCmd, queue_size = 10)

        self.seg_in = Image()
        self.odom = Odometry()
        self.road_mask = None
        self.color_in = Image()
        self.cmd_vel = VelCmd()
        self.union = None
        self.overlap = None
        self.IOU = 1
        self.reset_count = 0

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
        self.actions = [(1, -0.5), (1, 0), (1, 0.5)]
        self.preferences = np.array([0, 0, 0])
        self.alpha = 1 
        self.incremental_reward = 0
        self.step = 0

    def odom_cb(self, msg):
        if msg.pose.pose.position.z > 0.8:
            self.cmd_vel.twist.linear.z = -0.5
            #self.cmd_pub.publish(self.cmd_vel)
        else:
            self.cmd_vel.twist.linear.z = 0

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

    def calculate_reward(self):
        my_img = self.bridge.imgmsg_to_cv2(self.seg_in, desired_encoding = "passthrough")
        my_img = my_img[:,:,0].copy()

        # Masking out everything except road
        my_img[my_img != 246] = 0
        my_img[my_img == 246] = 255

        # Eroding to remove sparse outlier false positives, dilating to fill eroded gaps
        kernel = np.ones((2, 2))
        my_img = cv2.erode(my_img, kernel)
        kernel = np.ones((3, 3))
        my_img = cv2.dilate(my_img, kernel)

        if type(self.overlap) == type(None) or self.IOU <= 0.05:
            self.IOU = 1
            self.union = my_img.astype(np.bool_)
            self.overlap = my_img.astype(np.bool_)
        else:
            self.union += my_img.astype(np.bool_)
            self.union [self.union != 0] = 255
            self.overlap = self.overlap*(my_img.astype(np.bool_))
            self.IOU = self.overlap.sum()/float(self.union.sum())

        self.incremental_reward += self.IOU
        print("\nPreference list: ", self.preferences, "\n")

        return self.IOU

    def act(self):
        if self.step == 0:
            action_idx = random.randint(0, 2)
            self.cmd_vel.twist.linear.x = self.actions[action_idx][0]
            self.cmd_vel.twist.angular.z = self.actions[action_idx][1]
            self.cmd_pub.publish(self.cmd_vel)
        else:
            action_idx = np.argmax(self.preferences)
            self.cmd_vel.twist.linear.x = self.actions[action_idx][0]
            self.cmd_vel.twist.angular.z = self.actions[action_idx][1]
            self.cmd_pub.publish(self.cmd_vel)
        self.step += 1
        return action_idx

    def update_action_preferences(self, reward, action_idx):
        pref_softmax = np.exp(self.preferences)/(np.exp(self.preferences).sum())
        for idx, pref in enumerate(self.preferences):
            if action_idx == idx:
                self.preferences[idx] += self.alpha*(reward - self.incremental_reward)*(1 - pref_softmax[idx])
            else:
                try:
                    self.preferences[idx] += -self.alpha*(reward - self.incremental_reward)*pref_softmax[idx]
                except:
                    pdb.set_trace()

    def learn(self):
        self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)
        while not rospy.is_shutdown():
            action_idx = self.act()
            prev_time = rospy.get_time()
            while rospy.get_time() < prev_time + 1:
                action_idx = np.argmax(self.preferences)
                self.cmd_vel.twist.linear.x = self.actions[action_idx][0]
                self.cmd_vel.twist.angular.z = self.actions[action_idx][1]
                self.cmd_pub.publish(self.cmd_vel)
                pass

            reward = self.calculate_reward()
            self.update_action_preferences(reward, action_idx)
            print("IOU score: ", "{:.2f}".format(reward), "reset count: ", self.reset_count, end="         \r")
            if reward == 0:
                self.reset_env()

reward_class = reward_class()
reward_class.learn()
rospy.spin()
