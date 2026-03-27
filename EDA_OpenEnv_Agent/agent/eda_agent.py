import json
from openai import OpenAI

client = OpenAI()

def get_action(obs):

    prompt = f"""
    You are a data scientist.

    Dataset columns:
    {obs.columns}

    Previous steps:
    {obs.history}

    Choose next best EDA step:
    describe, missing, correlation, outliers, insight

    Return JSON:
    {{
        "action_type": "...",
        "parameters": {{}}
    }}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = res.choices[0].message.content

    try:
        return json.loads(text)
    except:
        return {"action_type": "describe", "parameters": {}}