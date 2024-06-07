
import skill_aware_panda_gym
import gymnasium as gym
env = gym.make('PandaSkillAware-v3', render_mode="rgb_array", friction=1.0, mass=1.0)

observation, info = env.reset()
states = []
for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    # info['state']  ∈ {'push', 'roll', 'pick','down'}
    states.append(info['state'])
    print(states)

    if terminated or truncated:
        observation, info = env.reset()
        if 'pick' in states: print('pick skill')
        elif 'roll' in states: print('roll skill')
        else: print('push skill')
        states.clear()

env.close()
