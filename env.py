import random
import string
from pydantic import BaseModel
from typing import List, Optional

# Import our schemas from models.py
from models import (
    CertificateRequest, ActionDecision, BlockchainObservation,
    BlockchainAction, BlockchainReward
)

# OpenEnv typically expects a return object with these fields for step/reset
class EnvResult(BaseModel):
    observation: BlockchainObservation
    reward: float = 0.0
    done: bool = False
    info: dict = {}

class BlockchainEnv:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.current_step = 0
        self.gas_balance = 0.0
        self.pending_requests: List[CertificateRequest] = []
        self.last_error: Optional[str] = None
        self.gas_cost_per_mint = 0.1

    def _generate_mock_wallet(self, valid: bool) -> str:
        """Helper to generate realistic wallets to trick the agent."""
        chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=40))
        if valid:
            return f"0x{chars}"
        else:
            # Invalid wallets: might be missing '0x', or wrong length
            return random.choice([f"0x{chars[:30]}", f"1x{chars}", chars])

    def _generate_requests(self, batch_size: int = 3) -> List[CertificateRequest]:
        """Generates a mix of valid and invalid requests for the agent to process."""
        requests = []
        for i in range(batch_size):
            is_valid = random.choice([True, False])
            score = random.uniform(70.0, 99.0) if is_valid else random.uniform(40.0, 69.9)
            wallet = self._generate_mock_wallet(is_valid)
            
            requests.append(CertificateRequest(
                request_id=f"REQ-{random.randint(1000, 9999)}",
                student_name=f"Student_{i}",
                course_name="Intro to Web3",
                final_score=round(score, 2),
                wallet_address=wallet
            ))
        return requests

    def state(self) -> BlockchainObservation:
        """Returns the current state without advancing the environment."""
        return BlockchainObservation(
            pending_requests=self.pending_requests,
            gas_balance=round(self.gas_balance, 2),
            current_step=self.current_step,
            last_error=self.last_error
        )

    def reset(self) -> EnvResult:
        """Resets the environment for a new episode."""
        self.current_step = 0
        self.gas_balance = 2.0  # Enough for roughly 20 valid mints
        self.last_error = None
        self.pending_requests = self._generate_requests(batch_size=3)
        
        return EnvResult(observation=self.state())

    def step(self, action: BlockchainAction) -> EnvResult:
        """Processes the agent's action, updates state, and calculates reward."""
        self.current_step += 1
        self.last_error = None
        step_reward = 0.0
        correct_decisions = 0
        total_requests = len(self.pending_requests)

        # Create a dictionary of actions for easy lookup
        decision_map = {dec.request_id: dec.decision for dec in action.decisions}

        for req in self.pending_requests:
            # Ground truth validation logic
            is_valid_score = req.final_score >= 70.0
            is_valid_wallet = req.wallet_address.startswith("0x") and len(req.wallet_address) == 42
            should_mint = is_valid_score and is_valid_wallet

            agent_decision = decision_map.get(req.request_id)

            if agent_decision == "mint":
                if should_mint and self.gas_balance >= self.gas_cost_per_mint:
                    self.gas_balance -= self.gas_cost_per_mint
                    correct_decisions += 1
                else:
                    # Penalize for minting invalid certs or trying to mint without gas
                    step_reward -= 0.5 
            elif agent_decision == "reject":
                if not should_mint:
                    correct_decisions += 1
                else:
                    # Penalize for rejecting a valid student
                    step_reward -= 0.5
            else:
                self.last_error = f"Missing or invalid decision for {req.request_id}"

        # Calculate partial reward for this step (max 1.0 per step for perfect batch)
        if total_requests > 0:
            step_reward += (correct_decisions / total_requests)

        # Clamp reward to ensure it stays somewhat bounded
        step_reward = max(0.0, min(1.0, step_reward))

        # Check if episode is done
        done = self.current_step >= self.max_steps or self.gas_balance <= 0

        # Generate next batch if not done
        if not done:
            self.pending_requests = self._generate_requests(batch_size=3)
        else:
            self.pending_requests = []

        return EnvResult(
            observation=self.state(),
            reward=step_reward,
            done=done,
            info={"correct_decisions": correct_decisions, "total_batch": total_requests}
        )