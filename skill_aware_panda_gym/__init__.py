import os
from gymnasium.envs.registration import register

ENV_IDS = []
for task in ["Push"]:
    env_id = f"PandaSkillAware-v3"
    register(
        id=env_id,
        entry_point=f"skill_aware_panda_gym.envs:Panda{task}Env",
        max_episode_steps=50,
    )

    ENV_IDS.append(env_id)
