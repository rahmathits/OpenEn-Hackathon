"""
env/eda_env.py

OpenEnv-compliant EDA environment.
- Tasks have clear objectives and difficulty progression
- Grader is dynamic (input-sensitive, not fixed)
- Rewards are shaped and bounded in [0, 1]
- Episode boundaries are well-defined
"""

import random
import pandas as pd
from typing import Tuple, Dict, Any

from env.models import Observation, Action, Reward
from env.grader import grade_task
from tools.eda_tools import execute_action


# Maps each task to the action that completes it
TASK_ACTION_MAP = {
    "detect_missing":   "missing",
    "find_correlation": "correlation",
    "generate_insight": "insight",
}

TASKS = [
    {
        "name":       "detect_missing",
        "difficulty": "easy",
        "objective":  "Identify all columns with missing values in the dataset.",
    },
    {
        "name":       "find_correlation",
        "difficulty": "medium",
        "objective":  "Find the strongest correlation between any two numeric columns.",
    },
    {
        "name":       "generate_insight",
        "difficulty": "hard",
        "objective":  "Generate a meaningful insight string referencing actual data values and column names.",
    },
]


class EDAEnv:

    def __init__(self, df: pd.DataFrame, max_steps: int = 8):
        self.df = df
        self.max_steps = max_steps
        self.history: list[str] = []
        self.steps: int = 0
        self.done: bool = False
        self.task: Dict[str, Any] | None = None
        self.cumulative_reward: float = 0.0
        self._last_result: Any = None

    # ─────────────────────────────────────────
    # 🔁 RESET
    # ─────────────────────────────────────────
    def reset(self) -> Observation:
        self.history = []
        self.steps = 0
        self.done = False
        self.cumulative_reward = 0.0
        self._last_result = None
        self.task = random.choice(TASKS).copy()
        return self._get_obs()

    # ─────────────────────────────────────────
    # 👀 OBSERVATION
    # ─────────────────────────────────────────
    def _get_obs(self) -> Observation:
        return Observation(
            dataset_head=self.df.head().to_dict(orient="records"),
            columns=list(self.df.columns),
            stats=self.df.describe().to_dict(),
            history=self.history.copy(),
            task=self.task["name"],
        )

    # ─────────────────────────────────────────
    # 🎯 STEP
    # ─────────────────────────────────────────
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if self.done:
            return (
                self._get_obs(),
                Reward(score=0.0, feedback="Episode already done. Call reset().", is_penalty=False),
                True,
                {},
            )

        result = execute_action(self.df, action.action_type)
        self._last_result = result
        reward = self._compute_reward(action, result)

        self.history.append(action.action_type)
        self.steps += 1
        self.cumulative_reward = round(self.cumulative_reward + reward.score, 4)

        if self.steps >= self.max_steps:
            self.done = True

        return (
            self._get_obs(),
            reward,
            self.done,
            {
                "task": self.task["name"],
                "objective": self.task["objective"],
                "steps": self.steps,
                "cumulative_reward": self.cumulative_reward,
            },
        )

    # ─────────────────────────────────────────
    # 🧮 REWARD  (bounded [0, 1])
    # ─────────────────────────────────────────
    def _compute_reward(self, action: Action, result: Any) -> Reward:

        # ── Repeated action penalty ──────────────────────────────────────
        if action.action_type in self.history:
            return Reward(
                score=0.1,
                feedback=f"Repeated action '{action.action_type}' — try something new. (-penalty)",
                is_penalty=True,
            )

        expected_action = TASK_ACTION_MAP.get(self.task["name"])

        # ── Correct action for current task → call dynamic grader ────────
        if action.action_type == expected_action:
            grade, feedback = grade_task(
                task_name=self.task["name"],
                df=self.df,
                history=self.history + [action.action_type],   # include current action
                result=result,
            )
            if grade >= 1.0:
                self.done = True
                return Reward(score=1.0, feedback=f"✅ Task complete! {feedback}", is_penalty=False)

            return Reward(score=round(grade, 4), feedback=f"📊 {feedback}", is_penalty=False)

        # ── Wrong action for the task (pipeline order is fine though) ────
        return Reward(
            score=0.2,
            feedback=f"Action '{action.action_type}' is valid but not relevant to task '{self.task['name']}' (expected '{expected_action}').",
            is_penalty=False,
        )

    # ─────────────────────────────────────────
    # 📊 STATE  (public API)
    # ─────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        return {
            "task":               self.task["name"] if self.task else None,
            "objective":          self.task["objective"] if self.task else None,
            "difficulty":         self.task["difficulty"] if self.task else None,
            "steps":              self.steps,
            "max_steps":          self.max_steps,
            "history":            self.history.copy(),
            "cumulative_reward":  self.cumulative_reward,
            "done":               self.done,
        }