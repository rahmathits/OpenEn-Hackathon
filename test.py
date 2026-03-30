import pandas as pd
from env.eda_env import EDAEnv
from env.models import Action

df  = pd.read_csv("sample_sales_data.csv")
env = EDAEnv(df, max_steps=10)
obs = env.reset()

done = False
while not done:
    # Replace this with your agent's policy
    action = Action(action_type="clean_data")
    obs, reward, done, info = env.step(action)
    print(f"reward={reward.score:.4f} | feedback={reward.feedback}")