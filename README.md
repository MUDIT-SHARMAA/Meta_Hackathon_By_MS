---
title: Blockchain Certificate Admin
emoji: ⛓️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv: Blockchain Certificate Administrator

## 🌍 Environment Description & Motivation
The **Blockchain Certificate Administrator** environment simulates a real-world Web3 operations task. As educational institutions move toward blockchain-based credentialing, automated agents are needed to act as smart contract gatekeepers. 

Instead of a toy game, this environment challenges the agent to process messy, real-world student data, strictly enforce business logic (grades >= 70, valid 42-character 0x wallet addresses), and manage a finite resource (gas balance) to successfully mint certificates without wasting funds.

## 📊 Observation and Action Spaces

### Observation Space
The agent observes the current state of the blockchain administration queue:
* `pending_requests` (List): A batch of student requests containing `request_id`, `student_name`, `course_name`, `final_score`, and `wallet_address`.
* `gas_balance` (Float): The remaining gas in the administrative wallet.
* `current_step` (Int): The current episode step.
* `last_error` (String | Null): Feedback if the previous JSON action was malformed.

### Action Space
The agent must output a strictly typed JSON list of decisions:
* `decisions` (List): Contains objects with `request_id`, `decision` (strictly "mint" or "reject"), and a brief `reason`.

## 🏆 Task Descriptions & Difficulty
1. **Easy (`easy_mint`)**: The agent must successfully read the state and execute basic 'mint' actions without formatting errors, achieving at least a 30% success rate.
2. **Medium (`medium_filter`)**: The agent must accurately filter out invalid requests (e.g., scores < 70, malformed wallets like missing '0x' or wrong length).
3. **Hard (`hard_gas_management`)**: The agent must process consecutive batches of requests perfectly. Any invalid mints waste gas, which causes early termination and a score of 0.0.

## 🚀 Setup and Usage Instructions

**1. Install Dependencies**
```bash
pip install -r requirements.txt