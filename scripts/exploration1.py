#!/usr/bin/env python3

# code to build a ROS node

# import all necessary packages
import math
import numpy as np
import typing as T

# from utils import plot_line_segments

import rclpy
from rclpy.node import Node
from scipy import linalg
from scipy.interpolate import splrep, splev

from asl_tb3_lib.control import BaseController
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D

from std_msgs.msg import Bool

from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw

from nav_msgs.msg import OccupancyGrid

from scipy.signal import convolve2d

# create Node 
class Explorer(Node):

    def __init__(self) -> None:
    # give it a default node name
        super().__init__("explorer")

        self.state = None
        self.map_data = None
        self.map_res = None
        self.map_origin = None
        self.current_goal = None  # np.array([x, y]) in world frame
        self.image_detected = False
        self.timer_active = False

        

        # Publisher
        self.goal_pub = self.create_publisher(
                TurtleBotState,
                "/cmd_nav",
                10
            )
                
        self.create_subscription(
            TurtleBotState,
            "/state",
            self.state_cb,
            10,
        )
        self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_cb,
            10,
        )

        self.perc_sub = self.create_subscription(Bool, "/detector_bool", self.perc_callback, 10)
        
        self.timer = self.create_timer(2.0, self.timer_cb)

        
        self.get_logger().info("Explorer node initialized, publishing goals to /cmd_nav")

    def perc_callback(self, msg : Bool) -> None:
        if msg.data: #detects stop sign
            self.get_logger().info(
            f"stop sign detected"
            )
            self.image_detected = True
            goal_msg = TurtleBotState()
            goal_msg = self.state
            self.goal_pub.publish(goal_msg)
            
            self.timer = self.create_timer(5.0, self.stop_callback)
            self.timer_active = True

        elif not msg.data:
            self.image_detected = False

    def stop_callback(self) -> None:
        self.timer.cancel()
        self.current_goal = None
        




    def timer_cb(self):
        if self.state is None:
            self.get_logger().warn("[Explorer] No state yet, skipping timer")
            return
        if self.map_data is None:
            self.get_logger().warn("[Explorer] No map yet, skipping timer")
            return

        # 1) Check if we’ve reached the current goal
        if self.current_goal is not None:
            gx, gy = self.current_goal
            dist = math.hypot(gx - self.state.x, gy - self.state.y)

            if dist < 0.1:  # 10 cm threshold
                self.get_logger().info(
                    f"[Explorer] Reached current goal at ({gx:.2f},{gy:.2f}), dist={dist:.3f}. "
                    "Clearing goal to select a new frontier."
                )
                self.current_goal = None
            else:
                # Still heading toward current goal; nothing to do this tick
                self.get_logger().debug(
                    f"[Explorer] Current goal ({gx:.2f},{gy:.2f}) still {dist:.3f} m away."
                )
                return

        # 2) If no current goal, pick a new frontier
        frontier_states = self.find_frontiers()
        if frontier_states.size == 0:
            self.get_logger().info("[Explorer] No frontiers found this tick")
            return

        goal_xy = self.choose_frontier(frontier_states, self.state)
        if goal_xy.size == 0:
            self.get_logger().warn("[Explorer] choose_frontier returned empty goal")
            return

        self.get_logger().info(
            f"[Explorer] Selecting new frontier goal at ({goal_xy[0]:.2f},{goal_xy[1]:.2f})"
        )
        self.send_goal(goal_xy, self.state)



    def state_cb(self, msg: TurtleBotState):
        self.state = msg
        self.get_logger().debug(
            f"[Explorer] State update: x={msg.x:.2f}, y={msg.y:.2f}, theta={msg.theta:.2f}"
        )

    def map_cb(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data, dtype=float).reshape((h, w))

        # ROS OccupancyGrid convention: 0=free, 100=occupied, -1=unknown
        self.map_data = data
        self.map_res = res
        self.map_origin = (origin_x, origin_y)

        self.get_logger().info(
            f"[Explorer] Map update: size=({w}x{h}), res={res:.3f}, "
            f"origin=({origin_x:.2f}, {origin_y:.2f})"
        )



    # def find_frontiers(self, occupancy: StochOccupancyGrid2D) -> np.ndarray:
    #     """
    #     Frontier Detection
    #     Returns an (N, 2) array of world-frame frontier positions.
    #     """
    #     window_size = 5    # defines the window side-length for neighborhood of cells to consider for heuristics
    #     ########################### Code starts here ###########################
    #     #pulling state info from occupancy
    #     unknown_states = (occupancy.probs == -1.0)
    #     known_states = (occupancy.probs != -1.0)

    #     occupied_states = (occupancy.probs >= occupancy.thresh)
    #     unoccupied_states = (occupancy.probs < occupancy.thresh)

    #     known_unoccupied_states = (known_states & unoccupied_states) #known and unoccupied states

    #     K = np.ones((window_size, window_size)) #ones grid of size window cell x window cell
    #     K[window_size//2, window_size//2] = 0  # don’t count the center cell

    #     unknown_cells = convolve2d(unknown_states, K, mode='same',boundary="fill", fillvalue=0) #sum the unknown cells in a window size grid around each cell (account for edges)
    #     known_cells = convolve2d(known_states, K, mode='same',boundary="fill", fillvalue=0) #sum the known cells
    #     occupied_cells = convolve2d(occupied_states, K, mode='same',boundary="fill", fillvalue=0) #sum the known cells in a window size grid around each cell
    #     unoccupied_cells = convolve2d(unoccupied_states, K, mode='same',boundary="fill", fillvalue=0) #sum the unoccupied cells

    #     known_unoccupied_cells = convolve2d(known_unoccupied_states, K, mode='same',boundary="fill", fillvalue=0) #sum the known and unoccupied cells
    #     frontier_mask = ((unknown_cells>=(0.2*(unknown_cells+known_cells)))& (occupied_cells==0)&(known_unoccupied_cells>=(0.3*(known_cells+unknown_cells)))) #mask to filter desired cells

    #     frontier_states = np.argwhere(frontier_mask) #returns true indices in frontier mask

    #     frontier_states = frontier_states[:, ::-1] #flip rows and columns to match graph

    #     frontier_states = occupancy.grid2state(frontier_states)  #scale to physical coordinates


    #     # ##finding closest frontier state
    #     # distances = np.linalg.norm(frontier_states-current_state, axis = 1) #L2 norm
    #     # min_index = np.argmin(distances) #minimum index
    #     # closest_distance = distances[min_index] #closest distance

    #     ########################### Code ends here ###########################

    #     # sanity check
    #     if np.isnan(frontier_states).any():
    #         self.get_logger().warn("find_frontiers: NaNs detected in frontier_states!")

    #     self.get_logger().debug(
    #         f"find_frontiers: frontier_states[0]={frontier_states[0]} "
    #         f" (total N={frontier_states.shape[0]})"
    #     )
    #     return frontier_states

    def find_frontiers(self) -> np.ndarray:
        """
        Frontier Detection from raw OccupancyGrid.
        Returns an (N, 2) array of world-frame frontier positions.
        """
        if self.map_data is None:
            self.get_logger().warn("find_frontiers: map_data is None")
            return np.empty((0, 2))

        probs = self.map_data   # (H, W)
        window_size = 12

        # Same semantics as before
        unknown_states = (probs == -1.0)
        known_states   = (probs != -1.0)

        # pick a threshold (typ. 50 for OccupancyGrid)
        thresh = 50.0
        occupied_states   = (probs >= thresh)
        unoccupied_states = (probs == 0)

        known_unoccupied_states = (known_states & unoccupied_states)

        K = np.ones((window_size, window_size))
        K[window_size // 2, window_size // 2] = 0

        unknown_cells        = convolve2d(unknown_states,        K, mode='same', boundary="fill", fillvalue=0)
        known_cells          = convolve2d(known_states,          K, mode='same', boundary="fill", fillvalue=0)
        occupied_cells       = convolve2d(occupied_states,       K, mode='same', boundary="fill", fillvalue=0)
        known_unocc_cells    = convolve2d(known_unoccupied_states, K, mode='same', boundary="fill", fillvalue=0)

        frontier_mask = (
            (unknown_cells >= 0.2 * (unknown_cells + known_cells)) &
            (occupied_cells == 0) &
            (known_unocc_cells >= 0.3 * (known_cells + unknown_cells))
        )

        num_frontier = int(np.count_nonzero(frontier_mask))
        self.get_logger().info(f"find_frontiers: {num_frontier} frontier cells")

        if num_frontier == 0:
            return np.empty((0, 2))

        frontier_idx = np.argwhere(frontier_mask)  # rows, cols

        # Convert grid indices → world coords
        origin_x, origin_y = self.map_origin
        res = self.map_res

        # col (j) -> x, row (i) -> y
        xs = origin_x + frontier_idx[:, 1] * res
        ys = origin_y + frontier_idx[:, 0] * res

        frontier_states = np.stack([xs, ys], axis=1)  # (N, 2)

        # sanity check
        if np.isnan(frontier_states).any():
            self.get_logger().warn("find_frontiers: NaNs detected in frontier_states!")

        self.get_logger().debug(
            f"find_frontiers: frontier_states[0]={frontier_states[0]} "
            f"(N={frontier_states.shape[0]})"
        )

        return frontier_states

    
    def choose_frontier(self, frontier_states: np.ndarray,
                        state: TurtleBotState) -> np.ndarray:
        """
        Choose closest frontier location to current robot pose
        """
        distances = np.linalg.norm(frontier_states-[state.x, state.y], axis = 1) #L2 norm
        min_index = np.argmin(distances) #minimum index

        goal_xy = frontier_states[min_index]

        self.get_logger().info(
            f"choose_frontier: picked index {min_index}, "
            f"goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), "
            # f"distance={min_dist:.2f} m from robot=({state.x:.2f}, {state.y:.2f})"
        )

        return frontier_states[min_index]
    
    def send_goal(self, goal_xy: np.ndarray, state: TurtleBotState):
        if goal_xy is None or goal_xy.size != 2 or state is None:
            self.get_logger().warn(f"send_goal called with bad inputs: goal_xy={goal_xy}, state={state}")
            return

        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        dx, dy = gx - state.x, gy - state.y
        theta = math.atan2(dy, dx)

        goal_msg = TurtleBotState()
        goal_msg.x = gx       # IMPORTANT: use frontier as goal
        goal_msg.y = gy
        goal_msg.theta = theta

        self.goal_pub.publish(goal_msg)
        self.current_goal = np.array([gx, gy])  # <--- remember it

        self.get_logger().info(
            f"send_goal: published goal -> x={gx:.2f}, y={gy:.2f}, theta={theta:.2f}"
        )



if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = Explorer()           # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits



