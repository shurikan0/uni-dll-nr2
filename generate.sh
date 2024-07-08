# Generate all motion planning demos for the dataset
for env_id in PlaceCube-v1
do
    python -m data_generation.run --env-id $env_id \
      --traj-name="trajectory" --only-count-success --save-video -n 1 \
      --shader="rt" # generate sample videos
    mv data/$env_id/motionplanning/0.mp4 data/$env_id/motionplanning/sample.mp4
    python -m data_generation.run --env-id $env_id --traj-name="trajectory" -n 1000 --only-count-success
done