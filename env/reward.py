def compute_reward(state, action, result):

    reward = 0
    feedback = []

    # completeness
    if "missing_checked" in state:
        reward += 0.2

    if action == "correlation":
        reward += 0.2
        feedback.append("correlation done")

    if action == "outliers":
        reward += 0.2

    # diversity
    if action not in state["history"]:
        reward += 0.2
    else:
        reward -= 0.1
        feedback.append("repeated action")

    # insight quality
    if action == "insight":
        if len(result) > 20:
            reward += 0.2
        else:
            reward -= 0.2

    return reward, ", ".join(feedback)

# from env.grader import grade_task

# def compute_reward(self, action, result):

#     reward = 0.0
#     feedback = []

#     # step penalty
#     reward -= 0.01

#     # grader score (0 → 1)
#     grade_score = grade_task(
#         self.task["name"],
#         self.df,
#         self.history,
#         result
#     )

#     # 🔥 convert grader → reward
#     reward += grade_score * 0.8

#     if grade_score > 0.8:
#         reward += 1.0
#         feedback.append("task success")
#         self.done = True

#     # repetition penalty
#     if action.action_type in self.history:
#         reward -= 0.1
#         feedback.append("repeated action")

#     return reward, ", ".join(feedback)