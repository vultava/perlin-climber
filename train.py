import numpy as np
import environment
import agent
import torch

# Hiperparametri ------------

MAP_DIM = 64
MAX_STEPS = 300
EPISODES = 10000
BATCH_SIZE = 64
TARG_NN_UPDATE_FREQ = 10

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

# ---------------------------
#
#
# Inicijalizacija mape i agenta ----------

map = environment.PerlinMap(
    map=np.zeros((MAP_DIM, MAP_DIM)),
    map_dim=MAP_DIM,
    climber_x=0,
    climber_y=0,
    final_x=0,
    final_y=0,
    steps=0,
    max_steps=MAX_STEPS
)

_ = map.create_map(4, 4, 0.5)


agent = agent.DQNAgent()
epsilon = EPSILON_START

# Glavna petlja za trening

for episode in range(1, EPISODES+1):

    _ = map.create_map(4, 2, 0.5)
    state = map.reset()

    total_rew = 0
    done = False

    while not done:
        action = agent.select_action(state, epsilon)

        next_state, reward, done, _ = map.step(action)

        agent.memory.push(state, action, reward, next_state, done)
        agent.learn(BATCH_SIZE)

        state = next_state
        total_rew += reward

    epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)

    if episode % TARG_NN_UPDATE_FREQ == 0:
        agent.update_target_nn()

    if episode % 10 == 0:
        print(f"Episode: {episode:4d}/{EPISODES} | Reward: {total_rew:7.2f} | Steps: {map.steps:3d} | Epsilon: {epsilon:.3f}")

    if episode in [100, 1000, 3000, 5000, 8000]:
        torch.save(agent.policy_nn.state_dict(), f"perlin_climber_dqn_64_withEuclid_v3_at{episode}.pth")
        print(f"Saved progress at episode {episode}!")


torch.save(agent.policy_nn.state_dict(), f"perlin_climber_dqn_64_withEuclid_v3_at{EPISODES}.pth")
