#!/usr/bin/env python3
import pdb
import rospy
import time
from sensor_msgs.msg import Image
from airsim_ros_pkgs.msg import VelCmd

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

        self.bridge = CvBridge()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def seg_cb(self, msg):
        self.seg_in = msg

    def color_cb(self, msg):
        self.color_in = msg

    def calculate_reward(self):
        my_img = self.bridge.imgmsg_to_cv2(self.seg_in, desired_encoding = "passthrough")
        my_img = my_img[:,:,0].copy()
        my_img[my_img != 246] = 0
        my_img[my_img == 246] = 255
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
            if self.IOU == 0:
                print("Resetting IOU again...")
                self.union = None 
                self.overlap = None
                self.client.reset()
                self.client.enableApiControl(True)
                self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)
                return self.IOU

        self.overlap = 255 * (self.overlap.astype(np.uint8))
        self.overlap[self.overlap != 0] = 255
        final_img = self.bridge.cv2_to_imgmsg(self.overlap, encoding = "mono8")
        self.img_pub.publish(final_img)
        return self.IOU

    def learn(self):
        self.seg_in = rospy.wait_for_message("/airsim_node/SimpleFlight/segmentation/Segmentation", Image, timeout=None)
        while not rospy.is_shutdown():
            reward = self.calculate_reward()
            print("IOU score: ", "{:.2f}".format(reward), end="    \r")
            

reward_class = reward_class()
reward_class.learn()
rospy.spin()
