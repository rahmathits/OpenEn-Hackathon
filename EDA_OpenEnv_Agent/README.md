# 🤖 EDA OpenEnv Agent

A **real-world Reinforcement Learning environment** built on the [OpenEnv](https://openenv.dev) standard, where an AI agent learns to perform **Exploratory Data Analysis (EDA)** on tabular datasets by following a structured data science pipeline.

---

## 📌 Problem Statement

Most RL environments are abstract — grids, games, simulations. This project models a **genuine data science workflow**: cleaning data, exploring it, engineering features, and training a model. The challenge for the agent is to complete these steps in the correct order and execute the right action for a given task, maximising cumulative reward across the episode.

The environment is designed so that **any agent** — rule-based, LLM-powered, or a trained RL model — can plug straight in via the standard `reset()` / `step()` / `state()` API.

---

## 🗂️ Project Structure

```
EDA_OpenEnv_Agent/
│
├── app.py                  # Streamlit dashboard (human-in-the-loop UI)
├── baseline_agent.py       # LLM baseline agent using OpenAI API
├── pipeline.py             # Pipeline ordering rules & reward shaping
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container for easy deployment
│
├── env/
│   ├── eda_env.py          # Core OpenEnv environment (reset/step/state)
│   ├── grader.py           # Dynamic task graders (input-sensitive scoring)
│   └── models.py           # Pydantic models: Observation, Action, Reward
│
└── tools/
    └── eda_tools.py        # Action executors (clean, eda, feature eng, train)
```

---

## 🔁 How the Environment Works

The environment follows the standard OpenEnv API:

```python
env = EDAEnv(df)
obs = env.reset()                        # sample a task, get initial observation

action = Action(action_type="clean_data")
obs, reward, done, info = env.step(action)  # execute action, get reward

state = env.state()                      # inspect full internal state
```

### Episode Flow

| Stage | What happens |
|---|---|
| `reset()` | A task is randomly sampled and the agent receives an observation of the dataset (head, columns, stats, history) |
| `step(action)` | The agent picks an action. The env executes it, the grader scores it, and a shaped reward is returned |
| Episode ends | When the task is completed (`reward = 1.0`) or `max_steps` is reached |

---

## 🗺️ Pipeline & Action Space

The agent must follow a **linear pipeline** before executing its task-specific action:

```
clean_data → eda → feature_engineering → train_model → [task action]
```

**Pipeline actions:**

| Action | Description |
|---|---|
| `clean_data` | Handle missing values, fix dtypes |
| `eda` | Summarise distributions, detect outliers |
| `feature_engineering` | Create or transform features |
| `train_model` | Fit a baseline model |

**Task-specific actions:**

| Task | Action | Difficulty |
|---|---|---|
| `detect_missing` | `missing` | Easy |
| `find_correlation` | `correlation` | Medium |
| `generate_insight` | `insight` | Hard |

---

## 🎯 Reward Design

All rewards are bounded in `[0, 1]`:

| Situation | Score | Note |
|---|---|---|
| In-order action — `clean_data` | `0.25` | First time only |
| In-order action — `eda` | `0.50` | First time only |
| In-order action — `feature_engineering` | `0.75` | First time only |
| In-order action — `train_model` | `1.00` | First time only |
| Correct task action (dynamic grader) | `0.20 – 1.00` | Based on actual data quality |
| Task complete | `1.00` | Grader returns full score |
| Wrong action for task | `0.20` | Valid but irrelevant |
| Repeated action | `0.10` | Penalty |
| Out-of-order (1 step skipped) | `0.25` | Penalty |
| Out-of-order (2 steps skipped) | `0.50` | Penalty |
| Out-of-order (3 steps skipped) | `0.75` | Penalty |

### Dynamic Graders

Each task is scored based on **actual data quality**, not a fixed value:

- **`detect_missing`** — scores by coverage: how many columns with missing values were found vs. total
- **`find_correlation`** — scores by strength: max correlation `≥ 0.9` → `1.0`, down to `0.2` for no signal
- **`generate_insight`** — scores across three dimensions: text length, numeric references, and column name mentions

---

## 🚀 Getting Started

### Local Setup

```bash
git clone https://github.com/your-repo/EDA_OpenEnv_Agent
cd EDA_OpenEnv_Agent

pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

### Run the LLM Baseline Agent

```bash
export OPENAI_API_KEY=sk-...

# Run baseline across all 3 tasks
python baseline_agent.py --csv your_data.csv

# Run with GPT-4o, 3 episodes per task for a stable average
python baseline_agent.py --csv your_data.csv --model gpt-4o --episodes 3 --steps 10
```

Results are saved automatically to `baseline_results.json`.

### Docker

```bash
# Build
docker build -t eda-openenv .

# Run Streamlit app
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... eda-openenv

# Run LLM baseline
docker run \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  eda-openenv \
  python baseline_agent.py --csv data/your_data.csv
```

---

## 🧠 Plugging In Your Own Agent

Any agent that speaks the `reset()` / `step()` / `state()` API works:

```python
import pandas as pd
from env.eda_env import EDAEnv
from env.models import Action

df  = pd.read_csv("your_data.csv")
env = EDAEnv(df, max_steps=10)
obs = env.reset()

done = False
while not done:
    # Replace this with your agent's policy
    action = Action(action_type="clean_data")
    obs, reward, done, info = env.step(action)
    print(f"reward={reward.score:.4f} | feedback={reward.feedback}")
```

---

## 📊 Baseline Results

The `baseline_agent.py` script runs a GPT-4o-mini agent across all 3 tasks and reports:

```
══ BASELINE SCORE SUMMARY ══════════════════════════════════
  Task                      Difficulty   Steps    Penalties  Avg Reward
  ─────────────────────────────────────────────────────────
  detect_missing            easy         6        0          0.7500
  find_correlation          medium       7        0          0.6800
  generate_insight          hard         8        1          0.5200
  ─────────────────────────────────────────────────────────
  BASELINE SCORE (mean across tasks)                        0.6500
════════════════════════════════════════════════════════════
```

---

## 🧩 Evaluation Criteria Mapping

| Criterion | Weight | How we address it |
|---|---|---|
| Real-world utility | 30% | Models a genuine EDA workflow on real uploaded CSV data |
| Task & grader quality | 25% | 3 tasks with difficulty progression; dynamic graders score actual data quality |
| Environment design | 20% | Clean state management, shaped rewards in `[0,1]`, proper episode boundaries |
| Code quality & spec | 15% | OpenEnv-compliant API, typed Pydantic models, Dockerfile, baseline script |
| Creativity & novelty | 10% | Pipeline ordering enforcement with graded penalties is an original mechanic |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** — interactive dashboard
- **Pydantic v2** — typed models for Observation, Action, Reward
- **OpenAI API** — LLM baseline agent
- **Pandas / NumPy / Scikit-learn** — EDA tooling
- **Docker** — containerised deployment

---

## 📄 License

MIT