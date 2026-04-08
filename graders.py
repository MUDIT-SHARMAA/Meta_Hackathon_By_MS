from typing import List, Any

def _calculate_base_score(trajectory: List[dict]) -> float:
    if not trajectory:
        return 0.01
    
    total_reward = sum(step.get("reward", 0.0) for step in trajectory)
    max_possible_reward = len(trajectory) * 1.0
    
    if max_possible_reward == 0:
        return 0.01
        
    score = total_reward / max_possible_reward
    # Clamp between 0.01 and 0.99 to satisfy "strictly between 0 and 1"
    return max(0.01, min(0.99, score))

def grade_easy(trajectory: List[dict], env_state: Any = None) -> float:
    score = _calculate_base_score(trajectory)
    # Return 0.95 for success instead of 1.0
    return 0.95 if score >= 0.3 else max(0.01, score)

def grade_medium(trajectory: List[dict], env_state: Any = None) -> float:
    score = _calculate_base_score(trajectory)
    return 0.95 if score >= 0.6 else max(0.01, score)

def grade_hard(trajectory: List[dict], env_state: Any = None) -> float:
    score = _calculate_base_score(trajectory)
    return 0.95 if score >= 0.9 else max(0.01, score)