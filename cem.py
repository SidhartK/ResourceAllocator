import numpy as np
from environment import ResourceAllocationEnv

debug = True

env = ResourceAllocationEnv()
n_iterations = 20
batch_size = 50
elite_frac = 0.2
n_elite = int(batch_size * elite_frac)

mean = np.zeros(env.action_space.shape)
std_dev = np.ones(env.action_space.shape) * (np.abs(env.action_space.high) + np.abs(env.action_space.low)) / 4


for iteration in range(n_iterations):
    if debug:
        print("\n" + "-" * 50)
        print(f"Iteration {iteration}, mean: {mean}, std_dev: {std_dev}")
    # Step 1: Sample batch actions from Gaussian policy
    actions = np.random.normal(mean, std_dev, (batch_size, env.action_space.shape[0]))
    # Clip action values to the range of the action space
    actions = np.clip(actions, env.action_space.low, env.action_space.high)
    
    # Step 2: Evaluate actions and store returns
    rewards = np.zeros(batch_size)
    for i in range(batch_size):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = actions[i]
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards[i] = total_reward
    
    # Step 3: Select elite actions
    elite_indices = rewards.argsort()[-n_elite:]
    elite_actions = actions[elite_indices]
    
    # Step 4: Update mean and standard deviation
    mean = elite_actions.mean(axis=0)
    std_dev = elite_actions.std(axis=0)
    
    print(f"Iteration {iteration}, mean reward: {rewards.mean()}")

env.close()
