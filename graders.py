from typing import List, Any

# A trajectory is typically passed as a list of dictionaries containing step data
# e.g., [{"reward": 0.5, "done": False, "info": {"correct_decisions": 2, "total_batch": 3}}, ...]

def _calculate_base_score(trajectory: List[dict]) -> float:
    """Helper function to calculate the average reward across all steps."""
    if not trajectory:
        return 0.0
    
    total_reward = sum(step.get("reward", 0.0) for step in trajectory)
    max_possible_reward = len(trajectory) * 1.0  # 1.0 is max reward per step
    
    if max_possible_reward == 0:
        return 0.0
        
    score = total_reward / max_possible_reward
    return max(0.0, min(1.0, score)) # Clamp between 0.0 and 1.0

def grade_easy(trajectory: List[dict], env_state: Any = None) -> float:
    """
    Easy Task: Did the agent manage to get at least a 30% success rate?
    This proves it understands the basic JSON schema and can make a 'mint' decision.
    """
    score = _calculate_base_score(trajectory)
    # If they hit 30% efficiency, we give them full marks for the easy task
    return 1.0 if score >= 0.3 else (score / 0.3)

def grade_medium(trajectory: List[dict], env_state: Any = None) -> float:
    """
    Medium Task: The agent needs to successfully filter out the bad wallets/scores.
    Requires at least 60% overall efficiency.
    """
    score = _calculate_base_score(trajectory)
    # Scales up to 1.0 based on hitting the 60% benchmark
    if score >= 0.6:
        return 1.0
    return max(0.0, score)

def grade_hard(trajectory: List[dict], env_state: Any = None) -> float:
    """
    Hard Task: Strict grading. 
    The agent must achieve near perfection (90%+) and not run out of gas 
    (which would result in a premature 'done' flag with low rewards).
    """
    score = _calculate_base_score(trajectory)
    
    # If the episode ended early because they wasted gas on bad mints, penalize heavily
    if len(trajectory) < 5 and trajectory[-1].get("reward", 0.0) < 0:
        return 0.0
        
    # Standard raw score, very hard to get 1.0 unless the agent is flawless
    return score