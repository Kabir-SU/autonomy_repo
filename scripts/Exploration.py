#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

from std_msgs.msg import Int64, Bool
from geometry_msgs.msg import Twist

class Exploration(BaseHeadingController):
    def __init__(self, node_name = "exploration"):
        
        # Call parent init
        super().__init__(node_name)
        self.declare_parameter('kp', 10.0)
        self.declare_parameter('active', True)
        self.image_detected = False
        
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.success_callback, 10)
        self.planned_path_sub = self.create_subscription(Path, "/planned_path", self.planned_path_callback, 10)
        self.smoothed_path_sub = self.create_subscription(Path, "/smoothed_path", self.smoothed_path_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(StochOccupancyGrid2D, "/map", self.map_callback, 10)
             
    def gay(self, msg:  Bool) -> None:
        if msg.data:
            self.image_detected = True
        else: 
            self.image_detected = False


    def explore(self, state: TurtleBotState, goal: TurtleBotState):
            """ returns potential states to explore
    Args:
        occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

    Returns:
        frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.
    """
    def explore(occupancy):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.
        """

        window_size = 13
        ########################### Code starts here ###########################

        frontier_states = []
        possible_moves = []

        for i in range(occupancy.size_xy[0]):
            for j in range(occupancy.size_xy[1]):
                if occupancy.probs[j][i] <= 0.5:
                    grid_xy = np.array([i, j])
                    state_xy = occupancy.grid2state(grid_xy)
                    possible_moves.append((state_xy, grid_xy))

        half_window = window_size // 2

        for state_xy, grid_xy in possible_moves:
            row_start = max(0, grid_xy[1] - half_window)
            row_end = min(occupancy.size_xy[1], grid_xy[1] + half_window + 1)
            col_start = max(0, grid_xy[0] - half_window)
            col_end = min(occupancy.size_xy[0], grid_xy[0] + half_window + 1)

            window = occupancy.probs[row_start:row_end, col_start:col_end]

            unknown_count = np.sum(window < 0)
            occupied_count = np.sum(window >= 0.5)
            unoccupied_count = np.sum((window >= 0) & (window < 0.5))

            n = window.size

            condition1 = unknown_count / n >= 0.2
            condition2 = occupied_count == 0
            condition3 = unoccupied_count / n >= 0.3

            if condition1 and condition2 and condition3:
                frontier_states.append(state_xy)

        return np.array(frontier_states)

    def desired_state(self):
        possible_states = self.explore(self.occupancy)
        return min(possible_states, key=lambda state: np.linalg.norm(state - np.array([self.state.x, self.state.y])))


if __name__ == "__main__":
    rclpy.init()
    node = Exploration()
    while not node.nav_success_sub:
        
    rclpy.spin(node)
    rclpy.shutdown()
