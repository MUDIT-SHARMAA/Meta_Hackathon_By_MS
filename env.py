import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
 
from models import BlockchainAction
from env import BlockchainEnv
 
# --- Mandatory Hackathon Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Swap to your active model
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")
TASK_NAME = os.getenv("MY_ENV_TASK", "hard_gas_management")
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
    success = False
 
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
 
    # Initialize Environment
    result = env.reset()
 
    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
 
            obs = result.observation
            user_prompt = f"Pending Requests: {obs.pending_requests}\nGas Balance: {obs.gas_balance}\nStep: {step}"
 
            try:
                # Call LLM
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,  # Deterministic logic preferred here
                )
                raw_response = (completion.choices[0].message.content or "").strip()
 
                # Parse JSON and execute action
                action_dict = json.loads(raw_response)
                action = BlockchainAction(**action_dict)
                result = env.step(action)
 
                reward = result.reward
                error = None
                action_str = json.dumps(action_dict).replace(" ", "")  # Compress for STDOUT
 
            except Exception as e:
                # If the LLM hallucinates or JSON parsing fails
                reward = 0.01  # Strictly above 0.0
                error = str(e).replace("\n", " ")
                action_str = "invalid_format"
                result.done = True  # End episode on fatal format error
 
            rewards.append(reward)
            steps_taken = step
 
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)
 
            if result.done:
                break
 
        # Calculate final normalized score — strictly between 0 and 1
        if steps_taken > 0:
            raw_score = sum(rewards) / steps_taken
        else:
            raw_score = 0.0
 
        # Clamp STRICTLY between 0 and 1 — never exactly 0.0 or 1.0
        score = max(0.01, min(0.99, raw_score))
        success = score >= 0.6
 
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
 
if __name__ == "__main__":
    main()