import argparse
from mani_skill.examples.motionplanning.panda import run 

# This will import the 'run' function (or module).

# Then use the function as it is intended in the example script, for instance:


#!python -m mani_skill.examples.motionplanning.panda.run -e "PegInsertionSide-v1"\
#    --num-traj 3000 \
#    --only-count-success \
#    --record-dir /content/drive/MyDrive/Data/Generated \
#    --traj-name traj3000 #File name is data100.h5
  
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PegInsertionSide-v1", help=f"Environment to run motion planning solver on. Available options")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=3000, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", default=True, help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="cpu", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, default="traj3000", help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="data", help="where to save the recorded trajectories")
    return parser.parse_args()
  
args = {
    "env_id": "PegInsertionSide-v1",
    "num_traj": 3000,
    "only_count_success": True,
    "record_dir": "/content/drive/MyDrive/Data/Generated",
    "traj_name": "traj3000"
}

run.main(parse_args())