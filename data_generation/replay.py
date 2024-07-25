import argparse
from mani_skill.trajectory.replay_trajectory import main


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", type=str, default="data/PegInsertionSide-v2/motionplanning/20240725_182655.h5")
    parser.add_argument(
        "-b",
        "--sim-backend",
        type=str,
        default="cpu",
        help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'",
    )
    parser.add_argument("-o", "--obs-mode", type=str, default="state_dict", help="target observation mode")
    parser.add_argument(
        "-c", "--target-control-mode", type=str, default="pd_joint_delta_pos", help="target control mode"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-traj", action="store_true", default=True, help="whether to save trajectories"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument(
        "--discard-timeout",
        action="store_true",
        help="whether to discard episodes that timeout and are truncated (depends on max_episode_steps parameter of task)",
    )
    parser.add_argument(
        "--allow-failure", action="store_true", help="whether to allow failure episodes"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--use-env-states",
        action="store_true",
        help="whether to replay by env states instead of actions",
    )
    parser.add_argument(
        "--use-first-env-state",
        action="store_true",
        help="use the first env state in the trajectory to set initial state. This can be useful for trying to replay \
            demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial \
            state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of demonstrations to replay before exiting. By default will replay all demonstrations",
    )

    parser.add_argument(
        "--reward-mode",
        type=str,
        help="specifies the reward type that the env should use. By default it will pick the first supported reward mode",
    )

    parser.add_argument(
        "--record-rewards",
        type=bool,
        help="whether the replayed trajectory should include rewards",
        default=False,
    )
    parser.add_argument(
        "--shader",
        default="default",
        type=str,
        help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer",
    )
    parser.add_argument(
        "--video-fps", default=30, type=int, help="The FPS of saved videos"
    )
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        type=str,
        help="The render mode used in the video saving",
    )

    return parser.parse_args()

#args = parse_args()
#args.traj_path = "data/PegInsertionSide-v1/motionplanning/traj3000.h5"  # Set the traj_path

main(parse_args())