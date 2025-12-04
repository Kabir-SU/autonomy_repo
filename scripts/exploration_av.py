#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from asl_tb3_msgs.msg import TurtleBotState      # CRITICAL
from asl_tb3_lib.grids import StochOccupancyGrid2D
from geometry_msgs.msg import Twist

import numpy as np
from scipy.signal import convolve2d
import time


class FrontierExplorerNode(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        # Parameters
        self.declare_parameter("occupancy_window_size", 13)
        self.declare_parameter("occupancy_threshold", 0.5)
        # self.declare_parameter("target_class", "stop sign")
        self.occupancy_window_size = int(self.get_parameter("occupancy_window_size").value)
        self.occupancy_threshold = float(self.get_parameter("occupancy_threshold").value)

        # Correct QoS for map: make it transient-local & reliable so
        # a latched/transient map published before this node starts
        # will still be received.
        map_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Internal state
        self.map_grid = None
        self.robot_x = None
        self.robot_y = None
        self.robot_theta = None
        self.nav_ready = True
        self._map_received_once = False
        self.exploration_stopped = False
        self.stop_timer = None

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos
        )

        self.state_sub = self.create_subscription(
            TurtleBotState,
            "/state",
            self.state_callback,
            10
        )

        self.nav_success_sub = self.create_subscription(
            Bool,
            "/nav_success",
            self.nav_success_callback,
            10
        )

        self.nav_active_sub = self.create_subscription(
            Bool,
            "/nav_active",
            self.nav_active_callback,
            10
        )

        self.detector_sub = self.create_subscription(
            Bool, 
            "/detector_bool", 
            self.stop_callback, 
            10)




# Publishers
        self.goal_pub = self.create_publisher(
            TurtleBotState,
            "/cmd_nav",
            10
        )

        self.stop_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )
        

        # Allow system to initialize for a few seconds
        self.startup_timer = self.create_timer(3.0, self.startup_callback)
        self.startup_fired = False

        self.get_logger().info("Frontier Explorer initialized. Waiting for map and pose...")

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------

    def startup_callback(self):
        """Wait a bit for map & state to stabilize before first explore."""

        if self.robot_x is not None and self.map_grid is not None:
            self.get_logger().info("Startup complete. Beginning exploration.")
            self.startup_timer.cancel()
            self.startup_fired = True
            self.try_explore()


    def map_callback(self, msg: OccupancyGrid):
        try:
            res = msg.info.resolution
            size = np.array([msg.info.width, msg.info.height])
            origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])

            data = np.array(msg.data, dtype=float).reshape((msg.info.height, msg.info.width))
            probs = data.copy()
            probs[probs == 100] = 1.0
            probs[probs == 0] = 0.0

            self.map_grid = StochOccupancyGrid2D(
                resolution=res,
                size_xy=size,
                origin_xy=origin,
                window_size=self.occupancy_window_size,
                probs=probs,
                thresh=self.occupancy_threshold
            )

            if not self._map_received_once:
                self.get_logger().info("Received first /map.")
                self._map_received_once = True

            # If we already have a robot pose, start exploration immediately
            # instead of waiting for the startup timer to fire.
            if not self.startup_fired and self.robot_x is not None:
                self.get_logger().info("Map and pose available. Beginning exploration.")
                self.startup_fired = True
                self.try_explore()

        except Exception as e:
            self.get_logger().error(f"map_callback error: {e}")

    def state_callback(self, msg: TurtleBotState):
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_theta = msg.theta
        # If we already have a map, start exploration immediately.
        if not self.startup_fired and self.map_grid is not None:
            self.get_logger().info("Pose and map available. Beginning exploration.")
            self.startup_fired = True
            self.try_explore()
     
    def nav_active_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Navigator active (planning).")
            self.nav_ready = False

    def nav_success_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Navigator: goal reached.")
        else:
            self.get_logger().warn("Navigator: goal failed.")
        self.nav_ready = True
        self.try_explore()

    def stop_callback(self, msg: Bool):
        if msg.data and self.nav_ready:
            self.get_logger().info("Stop sign detected. Stopping for 5 seconds...")

            #STOP robot motion
            stop_cmd = Twist()
            self.stop_pub.publish(stop_cmd)
            
            self.nav_ready = False  # Stop navigation immediately
            
            # Cancel any existing timer
            if self.stop_timer is not None:
                self.stop_timer.cancel()
            
            # Create a one-shot timer that fires after 5 seconds
            self.stop_timer = self.create_timer(5.0, self.resume_after_stop)


    def resume_after_stop(self):
        """Called 5 seconds after stop sign detection"""
        self.get_logger().info("Resume exploration after stop.")
        self.stop_timer.cancel()  # This was a one-shot timer
        self.stop_timer = None
        self.nav_ready = True
        self.try_explore()  # Resume exploration
    # ----------------------------------------------------------------------
    # Exploration logic
    # ----------------------------------------------------------------------

    def try_explore(self):
        if self.exploration_stopped:
            self.get_logger().info("Finished Exploration.")
            return
        if self.map_grid is None:
            return
        if self.robot_x is None:
            return
        if not self.nav_ready:
            return

        target = self.find_frontier()
        if target is None:
            self.get_logger().info("No more frontier targets. Exploration complete.")
            #self.try_explore()
            return

        tx, ty = target

        # Publish TurtleBotState goal
        goal = TurtleBotState()
        goal.x = float(tx)
        goal.y = float(ty)
        #goal.w = 0.0

        self.nav_ready = False
        self.goal_pub.publish(goal)
        self.get_logger().info(f"Published goal: ({tx:.2f}, {ty:.2f})")

    def explore(self, occupancy: StochOccupancyGrid2D):
        if occupancy is None:
            return np.zeros((0, 2))

        try:
            probs = occupancy.probs
            unknown = (probs == -1).astype(int)
            occupied = (probs >= 0.5).astype(int)
            known = ((probs >= 0) & (probs < 0.5)).astype(int)

            w = max(3, self.occupancy_window_size)
            kernel = np.ones((w, w))

            unknown_frac = convolve2d(unknown, kernel, mode="same") / (w * w)
            occupied_frac = convolve2d(occupied, kernel, mode="same") / (w * w)
            known_frac = convolve2d(known, kernel, mode="same") / (w * w)

            mask = (unknown_frac >= 0.2) & (occupied_frac == 0) & (known_frac >= 0.3)
            idx = np.argwhere(mask)

            frontiers = []
            for r, c in idx:
                try:
                    xy = occupancy.grid2state(np.array([c, r]))
                    frontiers.append(xy)
                except Exception:
                    pass

            if len(frontiers) == 0:
                self.exploration_stopped = True
                return np.zeros((0, 2))

            return np.vstack(frontiers)

        except Exception as e:
            self.get_logger().error(f"explore() error: {e}")
            return np.zeros((0, 2))

    def find_frontier(self):
        if self.map_grid is None:
            return None

        F = self.explore(self.map_grid)
        if F.shape[0] == 0:
            return None

        robot_xy = np.array([self.robot_x, self.robot_y])

        d = np.linalg.norm(F - robot_xy, axis=1)
        idx = np.argmin(d)
        sorted_d = np.argsort(d)
        for idx in sorted_d:
            target_xy = F[idx]
            cell = self.map_grid.state2grid(target_xy)
            if self.map_grid.is_free(cell):
                return target_xy
        self.get_logger().info("No valid cells.")
        return None

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
