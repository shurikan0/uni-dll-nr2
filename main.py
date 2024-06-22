import argparse

def main(on_cpu: bool, is_headless: bool, on_server: bool, on_amd: bool):
    print("cpu:", on_cpu)
    print("headless:", is_headless)
    print("server:", on_server)
    print("amd:", on_amd)

    # Make environment with task_1
    # Train model_1 on task with diffusion policy
    # Train model_2 on task without diffusion policy
    # Validate model_1
    # Validate model_2
    # Save model_1
    # Save model_2
    # Make environment with task_2
    # Validate model_1 on task_2
    # Validate model_2 on task_2
    # Compare model_1 and model_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", dest="cpu", action="store_const", const=True, default=False)
    parser.add_argument("--headless", dest="headless", action="store_const", const=True, default=False)
    parser.add_argument("--server", dest="server", action="store_const", const=True, default=False)
    parser.add_argument("--amd", dest="gpu", action="store_const", const=True, default=False)
    args = parser.parse_args()
    main(args.cpu, args.headless, args.server, args.gpu)
