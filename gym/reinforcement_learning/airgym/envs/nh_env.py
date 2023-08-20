import pdb
import setup_path
import airsim
import numpy as np
import math
import random
import time
from argparse import ArgumentParser

import cv2
import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimNHEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.random_poses = [
                #(0, -1, 0)
                (0, -1, 0, 0, 0, 0),
                (0, -1, 0, 0, 0, np.pi/2),
                (0, -1, 0, 0, 0, np.pi),
                (0, -1, 0, 0, 0, -np.pi/2),

                #(80.5, -1, 0)
                (80.5, -1, 0, 0, 0, 0),
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
        self.prev_img = None
        self.initial_yaw = 0
        self.forward_vel = 0
        self.lateral_vel = 0
        self.epsilon_vel = 0.000001
        self.yaw_vel = 0
        self.latch = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.reset_time = time.time()

        self.state = {
            "collision": False,
            "prev_position": np.zeros(3),
            "image": None,
        }

        self.drone = airsim.MultirotorClient(ip=ip_address, port=41455)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )

    def __del__(self):
        self.drone.reset()

    def quaternion_to_euler_angle_vectorized1(self, w, x, y, z):
        ysqr = y * y
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2>+1.0,+1.0,t2)
        #t2 = +1.0 if t2 > +1.0 else t2
        
        t2 = np.where(t2<-1.0, -1.0, t2)
        #t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)
        
        return X, Y, Z

    def _setup_flight(self):
        self.reset_time = time.time()
        self.drone.reset()
        self.latch = 0

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
        self.initial_yaw = np.array([self.random_poses[rndm_i][3]])
        self.drone.simSetVehiclePose(pose, True, "SimpleFlight")

        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
#        self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        #self.drone.moveToPositionAsync(0, -1, -2, 10).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        color_img_bytes = self.drone.simGetImage("color", airsim.ImageType.Scene)
        color_img_raw = np.frombuffer(color_img_bytes, dtype=np.uint8)
        color_img_np = cv2.imdecode(color_img_raw, -1)


        quad_info = self.drone.getMultirotorState().kinematics_estimated
        quad_vel = quad_info.linear_velocity
        quad_vel_ang = quad_info.angular_velocity
        quad_orientation = quad_info.orientation
        quad_orientation_euler = self.quaternion_to_euler_angle_vectorized1(quad_orientation.w_val, quad_orientation.x_val, quad_orientation.y_val, quad_orientation.z_val)
        effective_yaw = quad_orientation_euler[-1] - self.initial_yaw
        self.forward_vel = quad_vel.x_val*math.cos(effective_yaw)
        self.lateral_vel = quad_vel.y_val*math.cos(effective_yaw)
        self.yaw_vel = quad_vel_ang.z_val


        self.state["velocity"] = np.array([self.yaw_vel])
        self.state["image"] = color_img_np

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        #if self.latch == 0:
        #    forward_coeff = 2
        #    self.latch = 1
        #else:
        #    forward_coeff = 0
        #self.epsilon_vel *= -1

        #quad_offset_final = (quad_offset[0], quad_offset[1]+self.epsilon_vel, quad_offset[2])
        #print(quad_offset_final)

        #quad_info = self.drone.getMultirotorState().kinematics_estimated
        #quad_vel = quad_info.linear_velocity
        #qvl = [quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]
        #self.drone.moveByVelocityAsync(-qvl[0], -qvl[1], -qvl[2], 0.0, 5).join()
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            2,
        ).join()
        #self.drone.moveByRollPitchYawrateZAsync(
        #    quad_offset_final[0],
        #    quad_offset_final[1],
        #    quad_offset_final[2],
        #    z=0.0,
        #    duration=3
        #)


    def _compute_reward(self):
        img_bytes = self.drone.simGetImage("segmentation", airsim.ImageType.Segmentation)
        img_raw = np.frombuffer(img_bytes, dtype=np.uint8)
        img_np = cv2.imdecode(img_raw, -1)[:,:,0]
        road_mask = img_np.copy()
        road_id = 246
        road_mask[road_mask != road_id] = 0
        road_mask[road_mask == road_id] = 1
        mask_thresh = 0.1
        if np.count_nonzero(road_mask) < mask_thresh*road_mask.shape[0]*road_mask.shape[1]:
            return -50, 1 # Went out of road
        if type(self.prev_img) == type(None):
            self.prev_img = road_mask
            return 0, 0
        if self.state["collision"]:
            return -100, 1

        ideal_dist = np.zeros(road_mask.shape)
        points = np.array([[ideal_dist.shape[1], ideal_dist.shape[0]], [0, ideal_dist.shape[0]], [int(ideal_dist.shape[1]/2), int(ideal_dist.shape[0]/2)]])
        cv2.fillPoly(ideal_dist, pts=[points], color=1)
        
        dist_ideal = np.count_nonzero(ideal_dist, axis=0)
        dist_ideal = dist_ideal/dist_ideal.sum()
        dist_actual = np.count_nonzero(road_mask, axis=0)
        dist_actual = dist_actual/dist_actual.sum()
        indices = np.arange(dist_actual.shape[0])
        dist_bhatt = (dist_ideal*dist_actual)**0.5
        mean_bhatt = ((dist_bhatt*indices).sum())/(dist_bhatt.sum())
        reward = (dist_actual.shape[0] - int(abs(mean_bhatt - dist_ideal.shape[0]/2)))/100

        quad_info = self.drone.getMultirotorState().kinematics_estimated
        quad_vel = quad_info.linear_velocity
        quad_vel_ang = quad_info.angular_velocity
        self.yaw_vel = quad_vel_ang.z_val
        quad_orientation = quad_info.orientation
        quad_orientation_euler = self.quaternion_to_euler_angle_vectorized1(quad_orientation.w_val, quad_orientation.x_val, quad_orientation.y_val, quad_orientation.z_val)
        effective_yaw = quad_orientation_euler[-1] - self.initial_yaw
        self.forward_vel = quad_vel.x_val*math.cos(effective_yaw)

#        if int(time.time()-self.reset_time) > 3 and self.forward_vel < 0.01:
#            return -100, 1
#        reward -= self.forward_vel
        
        done = 0
        if reward <= -10:
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
#        if action == 0:
#            quad_offset = (self.step_length, 0, 0)
#        elif action == 1:
#            quad_offset = (0, self.step_length, 0)
#        elif action == 2:
#            quad_offset = (0, 0, self.step_length)
#        elif action == 3:
#            quad_offset = (-self.step_length, 0, 0)
#        elif action == 4:
#            quad_offset = (0, -self.step_length, 0)
#        else:
#            quad_offset = (0, 0, -self.step_length)
#        #elif action == 5:
#        #    quad_offset = (0, 0, -self.step_length)
#        #else:
#        #    quad_offset = (0, 0, 0)
        if action == 0:
            quad_offset = (0, self.step_length, -self.step_length)
        elif action == 1:
            quad_offset = (0, self.step_length, self.step_length)
        else:
            quad_offset = (0, self.step_length, 0)


        return quad_offset
