import sys
import os
import uvicorn
from fastapi import FastAPI

# Ensure the server can find your root models.py and env.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import BlockchainEnv
from models import BlockchainAction

app = FastAPI()
env = BlockchainEnv()

@app.post("/reset")
def reset_env():
    result = env.reset()
    return {"status": "ok", "observation": result.observation.model_dump()}

@app.post("/step")
def step_env(action: BlockchainAction):
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

@app.get("/state")
def get_state():
    return env.state().model_dump()

# The entry point the validator is looking for
# The entry point the validator is looking for
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()