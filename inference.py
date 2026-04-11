import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

from models import BlockchainAction
from env import BlockchainEnv

# --- Mandatory Hackathon Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") 
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")
# FIX 1: Match the exact ID from your openenv.yaml
TASK_NAME = os.getenv("MY_ENV_TASK", "easy_minting") 
BENCHMARK = "blockchain_certificate_admin"
MAX_STEPS = 5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a Smart Contract Administrator AI. 
    You will receive a list of student certificate requests and your current gas balance.
    Rules:
    1. Only 'mint' if final_score >= 70 AND wallet_address is exactly 42 chars starting with '0x'.
    2. Otherwise, 'reject'.
    3. Minting costs 0.1 gas. Do not mint if gas balance is insufficient.
    
    You MUST respond with valid JSON matching this schema exactly:
    {
      "decisions": [
        {"request_id": "REQ-1234", "decision": "mint", "reason": "Valid score and wallet"},
        {"request_id": "REQ-5678", "decision": "reject", "reason": "Invalid wallet length"}
      ]
    }
    Output ONLY JSON. No markdown formatting, no backticks.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = BlockchainEnv(max_steps=MAX_STEPS)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    result = env.reset()
    
    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            obs = result.observation
            user_prompt = f"Pending Requests: {obs.pending_requests}\nGas Balance: {obs.gas_balance}\nStep: {step}"
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0, 
                )
                raw_response = (completion.choices[0].message.content or "").strip()
                
                action_dict = json.loads(raw_response)
                action = BlockchainAction(**action_dict)
                result = env.step(action)
                
                reward = result.reward
                error = None
                action_str = json.dumps(action_dict).replace(" ", "") 
                
            except Exception as e:
                # FIX 2: Never output negative numbers or raw text! Bot parses this!
                reward = 0.01 
                error = "format_error" # No spaces allowed in error logs
                action_str = "{}" # Must be valid JSON brackets
                result.done = True 
                
            # FIX 3: Strictly clamp reward before printing to the terminal
            reward = max(0.01, min(0.99, reward))
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)
            
            if result.done:
                break
                
        score = sum(rewards) / steps_taken if steps_taken > 0 else 0.01
        score = max(0.01, min(0.99, score))
        success = score >= 0.6
        
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()