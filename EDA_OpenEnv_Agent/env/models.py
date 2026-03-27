from typing import List, Dict, Any
from pydantic import BaseModel

class Observation(BaseModel):
    dataset_head: List[Dict]
    columns: List[str]
    stats: Dict[str, Any]
    history: List[str]
    task: str   # 🔥 ADD THIS

class Action(BaseModel):
    action_type: str
    parameters: Dict[str, Any] = {}

class Reward(BaseModel):
    score: float
    feedback: str
    is_penalty: bool = False  # True when reward is an ordering violation