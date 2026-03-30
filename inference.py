"""
baseline_agent.py

Baseline inference script for EDA OpenEnv.

Uses the OpenAI API to run an LLM agent against the environment.
Reads credentials from environment variables.
Produces a reproducible baseline score across all 3 tasks.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline_agent.py --csv data/sample.csv
    python baseline_agent.py --csv data/sample.csv --model gpt-4o --episodes 3
"""

import os
import json
import argparse
import pandas as pd
from openai import OpenAI

from env.eda_env import EDAEnv, TASKS, TASK_ACTION_MAP
from env.models import Action
from pipeline import validate_action, apply_order_bonus, PIPELINE, get_completed_actions


# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
VALID_ACTIONS = ["clean_data", "eda", "feature_engineering", "train_model",
                 "missing", "correlation", "insight"]

SYSTEM_PROMPT = """You are an expert data science agent working inside an EDA (Exploratory Data Analysis) environment.

You will receive an observation describing a dataset and your current task.
Your job is to select the single best action to take next.

## Pipeline Rules (must follow in order):
1. clean_data
2. eda
3. feature_engineering
4. train_model

You cannot skip steps. After completing the pipeline, use the task-specific action.

## Task-specific actions:
- Task "detect_missing"   → use action: "missing"
- Task "find_correlation" → use action: "correlation"
- Task "generate_insight" → use action: "insight"

## Response format:
Respond with ONLY a JSON object, nothing else:
{"action": "<action_name>", "reason": "<one sentence why>"}

Valid actions: clean_data, eda, feature_engineering, train_model, missing, correlation, insight
"""


# ─────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────
class LLMAgent:

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "  export OPENAI_API_KEY=sk-..."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature  # 0.0 = deterministic, reproducible

    def select_action(self, obs, history: list) -> tuple[str, str]:
        """
        Ask the LLM to choose the next action given the current observation.
        Returns (action_type, reason).
        """
        completed = get_completed_actions(history)
        next_pipeline_step = next((s for s in PIPELINE if s not in completed), "pipeline complete")

        user_message = f"""## Current Observation

Task: {obs.task}
Columns: {obs.columns}
History (actions taken so far): {obs.history}

## Dataset Stats (first 5 rows):
{json.dumps(obs.dataset_head, indent=2)}

## Pipeline Status
Completed steps : {completed}
Next required   : {next_pipeline_step}

What is the single best action to take next?"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            action = parsed.get("action", "").strip()
            reason = parsed.get("reason", "")
            if action not in VALID_ACTIONS:
                print(f"  [warn] LLM returned unknown action '{action}', defaulting to pipeline step.")
                action = next_pipeline_step if next_pipeline_step != "pipeline complete" else "eda"
        except json.JSONDecodeError:
            print(f"  [warn] LLM response was not valid JSON: {raw!r}")
            action = next_pipeline_step if next_pipeline_step != "pipeline complete" else "eda"
            reason = "fallback due to parse error"

        return action, reason


# ─────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────
def run_episode(env: EDAEnv, agent: LLMAgent, task_override: dict | None = None, verbose: bool = True) -> dict:
    obs = env.reset()

    # Force a specific task for reproducible scoring across all 3 tasks
    if task_override:
        env.task = task_override.copy()
        obs = env._get_obs()

    history = []
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Task       : {obs.task}")
        print(f"  Objective  : {env.task['objective']}")
        print(f"  Difficulty : {env.task['difficulty']}")
        print(f"  Model      : {agent.model}")
        print(f"{'─' * 60}")

    while True:
        action_type, reason = agent.select_action(obs, history)

        # Pipeline validation
        penalty = validate_action(action_type, history)
        if penalty:
            reward = penalty
            done = False
            if verbose:
                print(f"  Step {step+1:02d} | {action_type:<22} | score={reward.score:.4f} | ⚠️  {reward.feedback}")
        else:
            action = Action(action_type=action_type)
            obs, reward, done, info = env.step(action)
            reward = apply_order_bonus(action_type, history, reward)
            if verbose:
                status = "✅ DONE" if done else "▶️ "
                print(f"  Step {step+1:02d} | {action_type:<22} | score={reward.score:.4f} | {status}")
                print(f"         reason : {reason}")
                print(f"         feedback: {reward.feedback}")

        history.append({
            "action":     action_type,
            "reward":     reward.score,
            "feedback":   reward.feedback,
            "is_penalty": penalty is not None,
            "done":       done,
        })
        total_reward += reward.score
        step += 1

        if done or step >= env.max_steps:
            break

    if verbose:
        print(f"{'─' * 60}")
        print(f"  Finished | steps={step} | total_reward={total_reward:.4f}")
        print(f"{'─' * 60}")

    return {
        "task":         env.task["name"],
        "difficulty":   env.task["difficulty"],
        "steps":        step,
        "total_reward": round(total_reward, 4),
        "history":      [h["action"] for h in history],
        "penalties":    sum(1 for h in history if h["is_penalty"]),
    }


# ─────────────────────────────────────────
# Main — runs all 3 tasks for reproducible baseline
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EDA OpenEnv — LLM Baseline Agent")
    parser.add_argument("--csv",      required=True,          help="Path to CSV dataset")
    parser.add_argument("--model",    default="gpt-4o-mini",  help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--steps",    type=int, default=10,   help="Max steps per episode (default: 10)")
    parser.add_argument("--episodes", type=int, default=1,    help="Runs per task for averaging (default: 1)")
    parser.add_argument("--quiet",    action="store_true",    help="Suppress per-step output")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    df = pd.read_csv("sample_sales_data.csv")
    print(f"\nDataset loaded : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Columns        : {list(df.columns)}")
    print(f"Model          : {args.model}")
    print(f"Episodes/task  : {args.episodes}")

    # ── Init ─────────────────────────────────────────────────────────────
    env   = EDAEnv(df, max_steps=args.steps)
    agent = LLMAgent(model=args.model, temperature=0.0)

    # ── Run all 3 tasks (reproducible baseline) ───────────────────────────
    all_results = []

    for task in TASKS:
        task_results = []
        print(f"\n{'═' * 60}")
        print(f"  Running task: {task['name']} (difficulty: {task['difficulty']})")
        print(f"{'═' * 60}")

        for ep in range(args.episodes):
            if args.episodes > 1:
                print(f"\n  Episode {ep + 1}/{args.episodes}")
            result = run_episode(env, agent, task_override=task, verbose=not args.quiet)
            task_results.append(result)

        avg_reward = sum(r["total_reward"] for r in task_results) / len(task_results)
        all_results.append({**task_results[-1], "avg_reward": round(avg_reward, 4)})

    # ── Baseline summary ─────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  BASELINE SCORE SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Task':<25} {'Difficulty':<12} {'Steps':<8} {'Penalties':<10} {'Avg Reward'}")
    print(f"  {'─'*55}")

    total_score = 0.0
    for r in all_results:
        print(f"  {r['task']:<25} {r['difficulty']:<12} {r['steps']:<8} {r['penalties']:<10} {r['avg_reward']:.4f}")
        total_score += r["avg_reward"]

    baseline_score = round(total_score / len(all_results), 4)
    print(f"  {'─'*55}")
    print(f"  {'BASELINE SCORE (mean across tasks)':<46} {baseline_score:.4f}")
    print(f"{'═' * 60}\n")

    # ── Write results to JSON for CI / judging ────────────────────────────
    output = {
        "model":          args.model,
        "dataset":        args.csv,
        "episodes_per_task": args.episodes,
        "task_results":   all_results,
        "baseline_score": baseline_score,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to baseline_results.json\n")


if __name__ == "__main__":
    main()