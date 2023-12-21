import gymnasium as gym
import panda_mujoco_gym
import pytest


def run_env(env):
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()
    # check that it allows to be closed multiple times
    env.close()


@pytest.mark.parametrize("env_id", panda_mujoco_gym.ENV_IDS)
def test_env(env_id):
    env = gym.make(env_id)
    run_env(env)
