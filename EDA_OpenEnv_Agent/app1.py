import streamlit as st
import pandas as pd
from env.eda_env import EDAEnv
from env.models import Action
import env

st.title("📊 EDA Agent (OpenEnv)")

file = st.file_uploader("Upload CSV")

# api_key = st.text_input(
#     "Enter your OpenAI API Key",
#     type="password",
#     placeholder="sk-..."
# )

# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key

if "history" not in st.session_state:
    st.session_state.history = []

# Add something to history
if st.button("Add Step"):
    st.session_state.history.append("New Step")

if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0

if file:
    df = pd.read_csv(file)

    # Initialize session state


# Display history
if st.session_state.history:
    for step in st.session_state.history:
        st.write(step)

    if "env" not in st.session_state:
        st.session_state.env = EDAEnv(df)
        st.session_state.obs = st.session_state.env.reset()

        obs = st.session_state.obs

    # if "history" not in st.session_state:
    #     st.session_state.history = []
        

    

    # st.write("Columns:", obs.columns)
    # st.write("History:", obs.history)

    # action = st.selectbox(
    #     "Action",
    #     ["describe", "missing", "correlation", "outliers", "insight"]
    # )

    # if "scores" not in st.session_state:
    #     st.session_state.scores = []

    # if "steps_data" not in st.session_state:
    #     st.session_state.steps_data = []

    # if st.button("Run Step"):
    #     action_obj = Action(action_type=action, parameters={})
    #     obs, reward, done, _ = st.session_state.env.step(action_obj)
    #     # obs, reward, done, info = st.session_state.env.step(Action(action_type=action))

    

    #     st.write("Reward:", reward.score)
    #     st.write("Feedback:", reward.feedback)

    #     st.session_state.obs = obs

    #     if done:
    #         st.success("Finished!")
    # # obs, reward, done, info = st.session_state.env.step(Action(action_type=action))
    

    # # store score
    #     st.session_state.scores.append(reward.score)

    #     # store step details
    #     st.session_state.steps_data.append({
    #         "Step": len(st.session_state.scores),
    #         "Action": action,
    #         "Score": round(reward.score, 3),
    #         "Cumulative Score": round(sum(st.session_state.scores), 3)
    #     })

    #     st.subheader("🏁 Total Score")

    #     total_score = sum(st.session_state.scores)

    #     st.metric("Total Reward", round(total_score, 3))      

    #     st.subheader("📋 Step-wise Performance")

    #     df_steps = pd.DataFrame(st.session_state.steps_data)

    #     st.dataframe(df_steps)

    #     if st.session_state.scores:
    #         best_step = max(st.session_state.scores)

    #         st.subheader("⭐ Best Step Score")
    #         st.write(best_step)

    # --------------------------------New Code Below--------------------------------##

    # -----------------------------
# Header
# -----------------------------
# st.set_page_config(page_title="OpenEnv Agent Dashboard", layout="wide")
# st.title("🤖 OpenEnv Agent Dashboard")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

action = st.sidebar.selectbox(
    "Choose Action",
    ["clean_data", "eda", "feature_engineering", "train_model"]
)

if st.sidebar.button("🔄 Reset Environment"):
    st.session_state.env.reset()
    st.session_state.history = []
    st.session_state.total_reward = 0
    st.success("Environment Reset")

if st.sidebar.button("▶️ Run Step"):
    action_obj = {"action": action}
    obs, reward, done, _ = st.session_state.env.step(action_obj)

    step_data = {
        "action": action_obj,
        "observation": obs,
        "reward": reward,
        "done": done
        }

    st.session_state.history.append(step_data)
    st.session_state.total_reward += reward

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

# -----------------------------
# Left Panel (Main Content)
# -----------------------------
with col1:

    # DATA / STATE OVERVIEW
    with st.expander("🔍 Current Environment State", expanded=True):
        if st.session_state.history:
            st.json(st.session_state.history[-1]["observation"])
        else:
            st.info("No steps yet. Run the agent.")

    # ACTION HISTORY
    with st.expander("📜 Step-by-Step Agent History"):
        if not st.session_state.history:
            st.warning("No history available")
        else:
            for i, step in enumerate(st.session_state.history):
                with st.expander(f"Step {i+1}: {step['action']['action']}"):
                    st.write("**Action:**", step["action"])
                    st.write("**Observation:**", step["observation"])
                    st.write("**Reward:**", step["reward"])
                    st.write("**Done:**", step["done"])

    # VISUALIZATION PLACEHOLDER
    with st.expander("📊 Visualizations"):
        st.info("Add EDA charts here (histograms, correlations, etc.)")

    # DEBUG PANEL
    with st.expander("🐞 Debug Panel"):
        st.write("Session State:")
        st.json(dict(st.session_state))

# -----------------------------
# Right Panel (Metrics)
# -----------------------------
with col2:

    with st.expander("🏆 Score Dashboard", expanded=True):
        st.metric("Total Reward", st.session_state.total_reward)
        st.metric("Steps Taken", len(st.session_state.history))

    with st.expander("🤖 Agent Status"):
        if st.session_state.history:
            last = st.session_state.history[-1]
            st.write("Last Action:", last["action"]["action"])
            st.write("Done:", last["done"])
        else:
            st.write("Agent not started")

    with st.expander("⚡ Quick Actions"):
        if st.button("Run 5 Steps Auto"):
            for _ in range(5):
                action_obj = {"action": action}
                obs, reward, done, _ = st.session_state.env.step(action_obj)

                step_data = {
                    "action": action_obj,
                    "observation": obs,
                    "reward": reward,
                    "done": done
                }

                st.session_state.history.append(step_data)
                st.session_state.total_reward += reward

                if done:
                    break

            st.success("Auto-run completed")