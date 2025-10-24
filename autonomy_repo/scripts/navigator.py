#!/usr/bin/env python3
import rclpy
import numpy as np
import scipy.interpolate 
from scipy.interpolate import splev
import typing as T

from rclpy.node import Node
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class Navigator(BaseNavigator):
    def __init__(self, node_name = "BaseNavigator", kp=1.0, kpx=1.0, kpy=1.0, kdx=1.0, kdy=1.0):
        super().__init__(node_name)
        self.kp = kp
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        
        self.V_PREV_THRESH = 0.0001
        self.t_prev = 0.0
        self.V_prev = 0.0
        self.om_prev = 0.0

    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState):
        #add gain initialization
        heading_error = wrap_angle(goal.theta - state.theta)
        om = self.kp * heading_error
        control_message = TurtleBotControl()
        control_message.omega = om
        return control_message
    
        
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:
        
        x, y, th = state.x, state.y, state.th

        dt = t - self.t_prev

        x_d = splev(t, plan.path_x_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        
        y_d = splev(t, plan.path_y_spline, der=0)
        yd_d = splev(t, plan.path_y_spline, der=1)
        ydd_d = splev(t, plan.path_y_spline, der=2)
        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < V_PREV_THRES:
            self.V_prev = V_PREV_THRES

        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                          [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########


        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        
        control = TurtleBotControl()
        control.V = V
        control.omega = om
        return control 
    

    def compute_trajectory_plan(self, 
                                state: TurtleBotState, 
                                goal: TurtleBotState,
                                occupancy: StochOccupancyGrid2D, 
                                resolution, 
                                horizon):
        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)
        astar = Astar((0, 0), (horizon, horizon), x_init, x_goal, occupancy, resolution)

        if not astar.solve() or len(astar.path < 4):
            return None

        self.t_prev = 0
        self.V_prev = 0

        path = np.asarray(astar.path)
        target_v = 0.5

        time_array = np.linspace(0, path.shape[0]/target_v, path.shape[0])

        path_x_spline = scipy.interpolate.splrep(time_array, path[:, 0], s=0.2)
        path_y_spline = scipy.interpolate.splrep(time_array, path[:, 1], s=0.2)

        return TrajectoryPlan(path=path, 
                            path_x_spline=path_x_spline,
                            path_y_spline=path_y_spline,
                            duration = time_array[-1])

        


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):  # runs automatically when new A* object is created
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

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
        if not (self.statespace_lo[0] < x[0] < self.statespace_hi[0] and
                self.statespace_lo[1] < x[1] < self.statespace_hi[1]):
            return False
        
        if self.occupancy.is_free(x) == True:
            return True
        else:
            return False
        # raise NotImplementedError("is_free not implemented")
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

        dist_euclidean = np.linalg.norm(np.array(x1) - np.array(x2))
        dist_manhattan = np.abs(x1[0]-x2[0])  + np.abs(x1[1] - x2[1])

        delta = np.array([x1[0]-x2[0], x1[1] - x2[1]])
        dist_maximum = max(np.abs(delta))

        return dist_maximum
    
        # raise NotImplementedError("distance not implemented")
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
        res_vector = [self.resolution, self.resolution]
        directions = [
            (-1, 1), (0,1), (1, 1),
            (-1, 0),          (1, 0),
            (-1, -1), (0, -1), (1, -1)
        ]

        neighbors_all = []
        for dx, dy in directions:
            neighbor_unresolved = (x[0]+ dx * self.resolution, x[1] + dy * self.resolution)
            neighbor = self.snap_to_grid(neighbor_unresolved)
            if self.is_free(neighbor):
                neighbors.append(neighbor)

        # raise NotImplementedError("get_neighbors not implemented")
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
        return list(reversed(path))

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

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
        while self.open_set:
            x_current = self.find_best_est_cost_through()

            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True

            self.open_set.remove(x_current)
            self.closed_set.add(x_current)

            for x in self.get_neighbors(x_current):
                if x in self.closed_set:
                    continue

                tentative_g = self.cost_to_arrive[x_current] + self.distance(x_current, x)

                if tentative_g < self.cost_to_arrive.get(x, float('inf')):
                    self.came_from[x] = x_current
                    self.cost_to_arrive[x] = tentative_g
                    self.est_cost_through[x] = tentative_g + self.distance(x, self.x_goal)

                    if x not in self.open_set:
                        self.open_set.add(x)

        return False
                

        # raise NotImplementedError("solve not implemented")
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

        

if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()