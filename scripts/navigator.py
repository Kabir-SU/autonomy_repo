#!/usr/bin/env python3

import rclpy

import numpy as np

import typing as T

from scipy.interpolate import splev, splrep
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState


class NavigatorNode(BaseNavigator):
    def __init__(self, kp=2.0, kpx=2.0, kpy=2.0, kdx=2.0, kdy=2.0) -> None:
        # give it a default node name
        super().__init__("navigator_node")
        self.kp = kp
        
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        
        self.V_PREV_THRESH = 0.0001
        self.t_prev = 0.0
        self.V_prev = 0.01
        self.om_prev = 0.0
        
    
    def compute_heading_control(self, curr_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:
        err = wrap_angle(desired_state.theta - curr_state.theta)
        w = self.kp * err
        new_control = TurtleBotControl()
        new_control.omega = w
        return new_control
        

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float,) -> TurtleBotControl:
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """
	
        x, y, th = state.x, state.y, state.theta
        dt = t - self.t_prev
        
        x_d = splev(t, plan.path_x_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        
        y_d = splev(t, plan.path_y_spline, der=0)
        yd_d = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)
        

        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRESH:
            self.V_prev = self.V_PREV_THRESH

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                          [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########

        # apply control limits
        # V = np.clip(V, -self.V_max, self.V_max)
        # om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
	
        control = TurtleBotControl()
        control.v = V
        control.omega = om
        return control
        
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
    
        statespace_lo = (occupancy.origin_xy[0] - horizon, occupancy.origin_xy[1] - horizon)
        statespace_hi = (statespace_lo[0] + (2*horizon), statespace_lo[1] + (2*horizon))
        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)
        astar = AStar(statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution)
    	
        if not astar.solve() or len(astar.path) < 4:
            return None
    	
        self.t_prev = 0.0
        self.V_prev = 0.01
        self.om_prev = 0.0
        
        path = np.asarray(astar.path)
        v_desired = 0.5

        path_diffs = np.diff(path, axis=0)
        path_distances = np.linalg.norm(path_diffs, axis=1)

        segment_times = path_distances/v_desired
        ts_unscaled = np.concatenate(([0], np.cumsum(segment_times)))
        
        # ts = np.linspace(0, path.shape[0]/v_desired, path.shape[0])
        path_x_spline = splrep(ts_unscaled, path[:, 0], s=0.05)
        path_y_spline = splrep(ts_unscaled, path[:, 1], s=0.05)
            
        return TrajectoryPlan(path=path,
    	path_x_spline=path_x_spline,
    	path_y_spline=path_y_spline,
    	duration=ts_unscaled[-1])
    	



        


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1, metric = 0):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.metric = metric

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states


    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x[0] < self.statespace_lo[0] or x[0] > self.statespace_hi[0] or x[1] < self.statespace_lo[1] or x[1] > self.statespace_hi[1]:
            return False
        return self.occupancy.is_free(np.array([x[0], x[1]]))
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        if self.metric == 0: return np.linalg.norm(np.array(x1) - np.array(x2))
        if self.metric == 1: return np.linalg.norm(np.array(x1) - np.array(x2), ord=1)
        if self.metric == 2: return np.linalg.norm(np.array(x1) - np.array(x2), ord=np.inf)
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for x_offset in [-self.resolution, 0, self.resolution]:
            for y_offset in [-self.resolution, 0, self.resolution]:
                if x_offset == 0 and y_offset == 0:
                    continue
                x_new = (x[0] + x_offset, x[1] + y_offset)
                x_new = self.snap_to_grid(x_new)

                if self.is_free(x_new):
                    neighbors.append(x_new)
        
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return np.array(list(reversed(path)))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        self.path = []
        while self.open_set:
            x_curr = self.find_best_est_cost_through()
            if x_curr == self.x_goal:
                while x_curr != self.x_init:
                    self.path.append(x_curr)
                    x_curr = self.came_from[x_curr]
                self.path.append(self.x_init)
                self.path = self.path[::-1]
                return True
            
            self.open_set.remove(x_curr)
            self.closed_set.add(x_curr)
            for x_neigh in self.get_neighbors(x_curr):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_curr] + self.distance(x_curr, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_curr
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False

        ########## Code ends here ##########

# class DetOccupancyGrid2D(object):
#     """
#     A 2D state space grid with a set of rectangular obstacles. The grid is
#     fully deterministic
#     """
#     def __init__(self, width, height, obstacles):
#         self.width = width
#         self.height = height
#         self.obstacles = obstacles

#     def is_free(self, x):
#         """Verifies that point is not inside any obstacles by some margin"""
#         for obs in self.obstacles:
#             if x[0] >= obs[0][0] - self.width * .01 and \
#                x[0] <= obs[1][0] + self.width * .01 and \
#                x[1] >= obs[0][1] - self.height * .01 and \
#                x[1] <= obs[1][1] + self.height * .01:
#                 return False
#         return True


       


if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = NavigatorNode()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits