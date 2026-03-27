import pandas as pd
from env.eda_env import EDAEnv
from env.models import Action
from agent.eda_agent import get_action

df = pd.read_csv("sample.csv")

env = EDAEnv(df)

obs = env.reset()
total_reward = 0

while True:

    action_dict = get_action(obs)
    action = Action(**action_dict)

    obs, reward, done, _ = env.step(action)

    print("Action:", action.action_type)
    print("Reward:", reward.score, reward.feedback)

    total_reward += reward.score

    if done:
        break

print("Final Score:", total_reward)