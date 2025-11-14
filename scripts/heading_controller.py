#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        
        # Call parent init
        super().__init__()
        self.kp = 10

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState):
        
        heading_error = wrap_angle(goal.theta - state.theta)
        correction_vel = self.kp * (heading_error)
        
        control_message = TurtleBotControl()
        control_message.omega = correction_vel
        
        # control_message.v = 10
        
        return control_message
    
if __name__ == "__main__":
    rclpy.init()
    node = HeadingController()
    rclpy.spin(node)
    rclpy.shutdown()