#!/usr/bin/env python3

import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):

    def __init__(self, node_name = "heading_controller"):
        super().__init__(node_name)
        # Proportional gain
        self.declare_parameter("kp", 2.0)


    @property
    def get_kp(self) -> float:
        return(self.get_parameter("kp")).value
    

    def compute_control_with_goal(
            self,
            current_state: TurtleBotState,
            desired_state: TurtleBotState) -> TurtleBotControl:
        proportional_error = desired_state.theta - current_state.theta
        desired_omega = self.get_kp * wrap_angle(proportional_error)
        message = TurtleBotControl()
        message.omega = desired_omega
        return message
    
if __name__ == "__main__":
    rclpy.init()
    sim_node = HeadingController()
    rclpy.spin(sim_node)
    rclpy.shutdown()