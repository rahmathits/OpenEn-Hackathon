from openai import OpenAI

def llm_insight_judge(insight: str, context: str, api_key: str) -> float:

    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are a strict data science evaluator.

    Dataset context:
    {context}

    Insight:
    {insight}

    Score this insight from 0 to 1 based on:
    - correctness
    - relevance
    - usefulness

    Return ONLY a number between 0 and 1.
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        score = float(res.choices[0].message.content.strip())
        return max(0.0, min(score, 1.0))

    except:
        return 0.0