import streamlit as st
import pandas as pd
from env.eda_env import EDAEnv
from env.models import Action
from Pipeline import validate_action, get_next_expected, PIPELINE

# -----------------------------
# Page Config
# -----------------------------
st.title("🤖 OpenEnv Agent Dashboard")

# -----------------------------
# About
# -----------------------------
with st.expander("📖 About this Project", expanded=True):
    st.markdown("""
    ### What is this?
    This is a **real-world Reinforcement Learning environment** built on the OpenEnv standard.
    An AI agent learns to perform **Exploratory Data Analysis (EDA)** on your dataset by interacting
    with the environment through a structured pipeline — just like a data scientist would.
 
    ### Problem Statement
    Most RL environments are abstract (grids, games, simulations). This environment models a **genuine
    data science workflow**: cleaning data, exploring it, engineering features, and training a model.
    The challenge for the agent is to complete these steps **in the right order** and **execute the
    correct action for the given task** — maximising its cumulative reward across the episode."""
 
    # ### How RL is implemented
    # The environment follows the standard OpenEnv API — `reset()`, `step()`, `state()` — so any agent
    # (rule-based, LLM, or trained RL model) can plug straight in. Each episode works like this:
 
    # | Stage | What happens |
    # |---|---|
    # | `reset()` | A task is sampled (`detect_missing`, `find_correlation`, or `generate_insight`) and the agent gets an observation of the dataset |
    # | `step(action)` | The agent picks an action. The environment executes it, scores the result, and returns a reward + next observation |
    # | Episode ends | When the task is completed (reward = 1.0) or max steps are reached |
 
    # ### Reward Design
    # Rewards are bounded between **0 and 1** and shaped to teach the agent good behaviour:
 
    # - ✅ **In-order action (first time)** → `+0.25` to `+1.00` scaled by pipeline stage
    # - 📊 **Correct task action** → scored dynamically by the grader based on actual data quality
    # - ⚠️ **Out-of-order action** → penalty (e.g. running `train_model` before `eda`)
    # - 🔁 **Repeated action** → small penalty to discourage inefficiency
 
    """###How to use this dashboard
    1. Upload a CSV file
    2. Watch the **Pipeline Progress** in the sidebar to know what step comes next
    3. Select an action and click **Run Step** — or let it run automatically
    4. Observe the reward and feedback at each step to understand what the agent learned
    """)

# -----------------------------
# File Upload
# -----------------------------
file = st.file_uploader("Upload CSV")

if file is not None:
    df = pd.read_csv(file)

    # Initialize environment once per uploaded file
    if "env" not in st.session_state:
        st.session_state.env = EDAEnv(df)
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.history = []
        st.session_state.total_reward = 0
else:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

# Pipeline progress
st.sidebar.markdown("### 🗺️ Pipeline Progress")
completed = [s["action"] for s in st.session_state.history if not s.get("is_penalty") and s["reward"] > 0]
for step in PIPELINE:
    if step in completed:
        st.sidebar.success(f"✅ {step}")
    elif get_next_expected(st.session_state.history) == step:
        st.sidebar.warning(f"▶️ {step}  ← next")
    else:
        st.sidebar.info(f"⬜ {step}")

st.sidebar.divider()

action = st.sidebar.selectbox(
    "Choose Action",
    ["clean_data", "eda", "feature_engineering", "train_model"]
)

if st.sidebar.button("🔄 Reset Environment"):
    st.session_state.env.reset()
    st.session_state.history = []
    st.session_state.total_reward = 0
    st.session_state.done = False
    st.success("Environment Reset")

if st.sidebar.button("▶️ Run Step"):
    if not st.session_state.done:
        action_obj = Action(action_type=action)

        # Check ordering before sending to env
        penalty = validate_action(action, st.session_state.history)
        if penalty:
            reward = penalty
            obs = st.session_state.history[-1]["observation"] if st.session_state.history else {}
            done = False
            st.sidebar.error(penalty.feedback)
        else:
            obs, reward, done, _ = st.session_state.env.step(action_obj)

        st.session_state.history.append({
            "action": action_obj.action_type,
            "observation": obs,
            "reward": reward.score,
            "feedback": reward.feedback,
            "is_penalty": reward.is_penalty or (penalty is not None),
            "done": done
        })
        st.session_state.total_reward += reward.score
        st.session_state.done = done
    else:
        st.warning("Episode is done. Reset the environment to continue.")

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

# -----------------------------
# Left Panel (Main Content)
# -----------------------------
with col1:

    with st.expander("🔍 Current Environment State", expanded=True):
        if st.session_state.history:
            st.json(st.session_state.history[-1]["observation"])
        else:
            st.info("No steps yet. Run the agent.")

    with st.expander("📜 Step-by-Step Agent History"):
        if not st.session_state.history:
            st.warning("No history available")
        else:
            for i, step in enumerate(st.session_state.history):
                is_penalty = step.get("is_penalty", False)
                label = f"⚠️ Step {i+1}: `{step['action']}`" if is_penalty else f"✅ Step {i+1}: `{step['action']}`"
                st.markdown(f"**{label}**")
                st.write("**Action:**", step["action"])
                st.write("**Observation:**", step["observation"])
                col_r, col_f = st.columns([1, 3])
                with col_r:
                    st.metric("Reward", f"{step['reward']:.2f}")
                with col_f:
                    if is_penalty:
                        st.error(step["feedback"])
                    else:
                        st.success(step["feedback"])
                st.write("**Done:**", step["done"])
                st.divider()

    with st.expander("📊 Visualizations"):
        st.info("Add EDA charts here (histograms, correlations, etc.)")

    with st.expander("🐞 Debug Panel"):
        st.write("Session State Keys:", list(st.session_state.keys()))

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
            st.write("Last Action:", last["action"])
            st.write("Done:", last["done"])
        else:
            st.write("Agent not started")

    with st.expander("⚡ Quick Actions"):
        if st.button("Run 5 Steps Auto"):
            for _ in range(5):
                if st.session_state.done:
                    st.warning("Episode ended before 5 steps.")
                    break

                action_obj = Action(action_type=action)
                penalty = validate_action(action, st.session_state.history)
                if penalty:
                    reward = penalty
                    obs = st.session_state.history[-1]["observation"] if st.session_state.history else {}
                    done = False
                else:
                    obs, reward, done, _ = st.session_state.env.step(action_obj)

                st.session_state.history.append({
                    "action": action_obj.action_type,
                    "observation": obs,
                    "reward": reward.score,
                    "feedback": reward.feedback,
                    "is_penalty": penalty is not None,
                    "done": done
                })
                st.session_state.total_reward += reward.score
                st.session_state.done = done

            else:
                st.success("Auto-run completed (5 steps)")