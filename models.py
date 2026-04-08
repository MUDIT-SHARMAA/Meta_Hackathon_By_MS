from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Nested Models for cleaner structure ---
class CertificateRequest(BaseModel):
    request_id: str
    student_name: str
    course_name: str
    final_score: float = Field(description="Score out of 100. Must be >= 70 to pass.")
    wallet_address: str = Field(description="Must start with '0x' and be exactly 42 characters long.")

class ActionDecision(BaseModel):
    request_id: str
    decision: Literal["mint", "reject"] = Field(description="Whether to mint the certificate or reject the request.")
    reason: str = Field(description="Short explanation of why it was minted or rejected.")

# ---------------------------------------------------------
# OBSERVATION SPACE (What the agent sees)
# ---------------------------------------------------------
class BlockchainObservation(BaseModel):
    pending_requests: List[CertificateRequest] = Field(
        description="List of student certificate requests awaiting processing."
    )
    gas_balance: float = Field(
        description="Remaining gas balance in the admin wallet to pay for minting."
    )
    current_step: int = Field(
        description="The current step in the episode."
    )
    last_error: Optional[str] = Field(
        default=None, 
        description="Error message if the previous action was invalid."
    )

# ---------------------------------------------------------
# ACTION SPACE (What the agent can do)
# ---------------------------------------------------------
class BlockchainAction(BaseModel):
    decisions: List[ActionDecision] = Field(
        description="The agent's decisions for the pending requests."
    )

# ---------------------------------------------------------
# REWARD SPACE (How the agent is scored)
# ---------------------------------------------------------
class BlockchainReward(BaseModel):
    score: float = Field(
        ge=0.0, le=1.0, 
        description="Normalized score between 0.0 and 1.0 based on correct mints and rejections."
    )
    feedback: str = Field(
        description="Feedback detailing gas wasted, incorrect mints, or missed valid certificates."
    )