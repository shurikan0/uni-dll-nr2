import numpy as np
import sapien

from .custom_envs.place_cube import PlaceCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PlaceCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    
    # Object manipulation adjustments
    obb = get_actor_obb(env.cube) 
    goal_pose = env.goal_site.pose 
    initial_offset = 0.05 # Move up 5cm from the current pose for a better approach

    # Get the current grasp pose (since the robot is already holding the cube)
    grasp_pose = env.agent.get_grasp_pose(env.cube)[0]  # Assuming a single grasp pose

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # Initial offset adjustment
    reach_pose = grasp_pose * sapien.Pose([0, 0, initial_offset])  
    planner.move_to_pose_with_screw(reach_pose)


    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(goal_pose) 

    # Release the cube
    planner.open_gripper()

    planner.close()
    return res