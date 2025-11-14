#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

from std_msgs.msg import Int64, Bool
from geometry_msgs.msg import Twist

class PerceptionController(BaseHeadingController):
    def __init__(self, node_name = "perception_controller"):
        
        # Call parent init
        super().__init__(node_name)
        self.declare_parameter('kp', 10.0)
        self.declare_parameter('active', True)
        self.image_detected = False
        
        self.detector_bool = self.create_subscription(Bool, "/detector_bool", self.gay, 10)
                
    def gay(self, msg:  Bool) -> None:
        if msg.data:
            self.image_detected = True
        else: 
            self.image_detected = False
            
    @property 
    def get_kp(self) -> float:
        return (self.get_parameter('kp').value)
    
    @property 
    def get_active(self) -> bool:
        return (self.get_parameter('active').value)

    def compute_control_with_goal(self, state: TurtleBotState, goal: TurtleBotState):
        
        #heading_error = wrap_angle(goal.theta - state.theta)
        #correction_vel = self.kp * (heading_error)
        
        control_message = TurtleBotControl()
        if not self.image_detected:
            control_message.omega = 0.2
        else:
            control_message.omega = 0.0
        
        # control_message.v =
        
        return control_message
    
if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()