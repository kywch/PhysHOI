import os
from dataclasses import dataclass

import isaacgym  # noqa
import torch
import tyro

from cleanrl_ppo import seed_everything, make_env, Agent


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_id: int = 0
    """the gpu id to use"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Env/sim-specific arguments
    env_id: str = "PhysHOI_BallPlay"
    """the id of the environment"""
    env_cfg_file: str = "physhoi/data/cfg/physhoi.yaml"
    """the path to the environment configuration file"""
    motion_file: str = "physhoi/data/motions/BallPlay/backdribble.pt"
    """the path to the motion file"""
    headless: bool = False
    """whether to run the environment in headless mode"""
    num_envs: int = 1
    """the number of parallel game environments"""

    physx_num_threads: int = 4
    """the number of cores used by PhysX"""
    physx_num_subscenes: int = 0
    """the number of PhysX subscenes to simulate in parallel"""
    physx_num_client_threads: int = 0
    """the number of client threads that process env slices"""

    # Checkpoint replay
    checkpoint: str = "tests/agent_04800.pth"
    """the path to the checkpoint file"""
    play_motion: bool = False
    """whether to play the input motion, instead of replaying the checkpoint"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed_everything(args)

    # env setup
    envs = make_env(args, use_gpu=False)
    device = envs.device

    # agent setup
    agent = Agent(envs).to(device)
    agent_weights = torch.load(args.checkpoint)
    agent.load_state_dict(agent_weights)

    # image save path
    motion_name = envs.task.motion_file[len("physhoi/data/motions/BallPlay/") : -3]
    image_folder = "tests/images/" + motion_name
    os.makedirs(image_folder, exist_ok=True)

    obs = envs.reset()
    done = False
    done_count = 0
    episode_return = 0
    for i in range(1, 300):
        action, logprob, _, value = agent.get_action_and_value(obs)

        if args.play_motion:
            envs.task.play_dataset_step(i)
            if i >= envs.task.max_episode_length:
                done_count = 3
        else:
            obs, reward, done, info = envs.step(action)
            print("Step", i, ", Reward", reward, ", Done", done)
            episode_return += reward

        # save images
        image_name = f"{image_folder}/image_{i:05d}.png"
        envs.gym.write_viewer_image_to_file(envs.viewer, image_name)

        if done:
            done_count += 1
            print("Episode return:", episode_return[0], ", Resetting...")
            obs = envs.reset()
            episode_return = 0

        if done_count >= 3:
            break
