from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    rviz_config = PathJoinSubstitution([FindPackageShare("autonomy_repo"), "rviz", "default.rviz"])
    
    navigator_node = Node(
        package='autonomy_repo',
        executable='navigator.py',
    )
    
    rviz_goal_relay = Node(
        package='asl_tb3_lib',
        executable='rviz_goal_relay.py',
        parameters=[{"output_channel": '/cmd_nav'}],
    )
    
    state_publisher = Node(
        package='asl_tb3_lib',
        executable='state_publisher.py',
        parameters=[{"use_sim_time": False}]
    )
    
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("asl_tb3_sim"),
                "launch",
                "rviz.launch.py"
            ])
        ),
        launch_arguments={
            'config': rviz_config,
            'use_sim_time': 'False'
        }.items(),
    )

    return LaunchDescription([
        rviz_goal_relay,
        state_publisher,
        navigator_node,
        rviz_launch,
    ])
