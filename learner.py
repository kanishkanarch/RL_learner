#!/usr/bin/env python3
import pdb

import rospy
from sensor_msgs.msg import Image
from airsim_ros_pkgs.msg import VelCmd

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
        self.img_pub = rospy.Publisher("/easytopic", Image, queue_size = 5)
        self.cmd_pub = rospy.Publisher("/airsim_node/SimpleFlight/vel_cmd_body_frame", VelCmd, queue_size = 10)

        self.seg_in = Image()
        self.road_mask = None
        self.color_in = Image()
        self.cmd_stop = VelCmd()
        self.union = None
        self.overlap = None
        self.IOU = 1
        self.reset_count = 0

        self.bridge = CvBridge()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.random_poses = [
                (0, -1, 0, 0, 0, 0),
                (0, -1, 0, 0, 0, np.pi/2),
                (0, -1, 0, 0, 0, np.pi),
                (0, -1, 0, 0, 0, -np.pi/2),

                (80.5, -1, 0, 0, 0, 0),
                (80.5, -1, 0, 0, 0, np.pi/2),
                (80.5, -1, 0, 0, 0, np.pi),
                (80.5, -1, 0, 0, 0, -np.pi/2),

                (128.5, -1, 0, 0, 0, np.pi/2),
                (128.5, -1, 0, 0, 0, np.pi),
                (128.5, -1, 0, 0, 0, -np.pi/2),

                (128.5, 126.5, 0, 0, 0, np.pi),
                (128.5, 126.5, 0, 0, 0, -np.pi/2),

                (0, 126.5, 0, 0, 0, 0),
                (0, 126.5, 0, 0, 0, np.pi),
                (0, 126.5, 0, 0, 0, -np.pi/2),

                (-127.5, 126.5, 0, 0, 0, 0),
                (-127.5, 126.5, 0, 0, 0, -np.pi/2),

                (-127.5, -1, 0, 0, 0, 0),
                (-127.5, -1, 0, 0, 0, np.pi/2),
                (-127.5, -1, 0, 0, 0, -np.pi/2),

                (-127.5, -128.5, 0, 0, 0, 0),
                (-127.5, -128.5, 0, 0, 0, np.pi/2),

                (0, -128.5, 0, 0, 0, 0),
                (0, -128.5, 0, 0, 0, np.pi/2),
                (0, -128.5, 0, 0, 0, np.pi),

                (80.5, -128.5, 0, 0, 0, 0),
                (80.5, -128.5, 0, 0, 0, np.pi/2),
                (80.5, -128.5, 0, 0, 0, np.pi),

                (128.5, -128.5, 0, 0, 0, np.pi/2),
                (128.5, -128.5, 0, 0, 0, np.pi),


]

    def seg_cb(self, msg):
        self.seg_in = msg

    def color_cb(self, msg):
        self.color_in = msg

    def observation(self):
        self.overlap = 255 * (self.overlap.astype(np.uint8))
        self.overlap[self.overlap != 0] = 255
        final_img = self.bridge.cv2_to_imgmsg(self.overlap, encoding = "mono8")
        self.img_pub.publish(final_img)

    def reset_env(self):
        self.reset_count += 1
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


        if type(self.overlap) == type(None) or self.IOU == 0:
            self.IOU = 1
            self.union = my_img.astype(np.bool_)
            self.overlap = my_img.astype(np.bool_)
        else:
            self.union += my_img.astype(np.bool_)
            self.union [self.union != 0] = 255
            self.overlap = self.overlap*(my_img.astype(np.bool_))
            self.IOU = self.overlap.sum()/float(self.union.sum())

        return self.IOU

    def learn(self):
        self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)
        while not rospy.is_shutdown():
            reward = self.calculate_reward()
            self.observation()
            print("IOU score: ", "{:.2f}".format(reward), "reset count: ", self.reset_count, end="         \r")
            if reward == 0:
                self.reset_env()


            

reward_class = reward_class()
reward_class.learn()
rospy.spin()
