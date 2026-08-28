# Small-LLM Airline Tool-Calling Workbench

This project fine-tunes Qwen3-1.7B with LoRA for structured airline customer-support tasks. It contains a local database, executable airline tools, a deterministic verifier, multi-turn rollouts, replayable RL tasks, and SFT data preparation.

## 1. Action parser and inference updation

The canonical tool contract lives in `Tool_calling/Tool_calling/airline_gym/schemas.py`.

The five public tools are:

- `search_flights`
- `get_booking_details`
- `book_reservation`
- `cancel_reservation`
- `update_reservation`

`action_parser.py`, `inference.py`, and `infer_airline_lora.py` use this contract. Model output must be exactly one JSON object. Supported actions are `tool_call`, `final_answer`, and `ask_user`.

Example tool call:

```json
{"action":"tool_call","tool_name":"cancel_reservation","arguments":{"reservation_id":"H0MVIE"}}
```

Legacy names such as `search_direct_flight` and `get_reservation_details` are rejected by the current parser.

## 2. Rollout loop implementation

The rollout implementation is in `Tool_calling/Tool_calling/airline_gym/rollout.py`.

The execution flow is:

```text
model output
  -> strict parse_action()
  -> valid action?
       no  -> structured failed rollout, reward 0.0
       yes -> tool call or final answer
  -> AirlineEnv.step(tool, arguments)
  -> append structured tool observation
  -> generate the next model action
  -> deterministic scorer
```

Example successful episode:

```yaml
User: "Yes, cancel reservation H0MVIE."

Model output:
  action: tool_call
  tool_name: cancel_reservation
  arguments:
    reservation_id: H0MVIE

Tool observation:
  ok: true
  reservation_id: H0MVIE
  status: cancelled

Model output:
  action: final_answer
  content: "Reservation H0MVIE has been cancelled."
```

The returned trajectory contains `status`, `messages`, `assistant_actions`, `tool_observations`, `call_log`, `final_answer`, and `score`.

Malformed JSON, mixed prose and JSON, unknown tools, invalid argument shapes, and missing final answers become structured failures instead of crashing a training batch. Normal business errors from the simulator remain visible as tool observations so the model can recover before the step limit.

The scorer in `Tool_calling/Tool_calling/airline_gym/scorer.py` evaluates parser validity, tool correctness, argument correctness, policy compliance, required facts, efficiency, and final database state.

## 3. Multi-turn RL task

An RL row describes a replayable episode, not only one SFT completion. Generated records come from `Tool_calling/Tool_calling/data/candidates_v2.jsonl` and are converted by `rl_dataset.py`.

Each generated RL row contains:

- `prompt`: initial system and user messages
- `task_id`: stable generated task identifier
- `split`: `train`, `valid`, or `test`
- `db_hash`: hash of the frozen simulator database
- `multi_turn`: whether the task contains multiple actions/turns
- `policy`: the airline policy text
- `tool_schemas`: the five canonical tool declarations
- `answer_key`: expected calls, required facts, allowed state changes, and expected final state hash

Example answer key:

```json
{
  "expected_calls": [
    {
      "name": "cancel_reservation",
      "arguments": {"reservation_id": "H0MVIE"}
    }
  ],
  "expected_state_hash": "...",
  "expected_new_ids": [],
  "required_facts": ["H0MVIE"]
}
```

The helper `replay_answer_key()` resets `AirlineEnv`, replays every expected call, and verifies the final state hash and newly created IDs.

## Build and validate RL data

Run from the `Small-LLM` directory:

```powershell
.\.venv\Scripts\python.exe rl_dataset.py
.\.venv\Scripts\python.exe validate_rl_tasks.py
```

When `Tool_calling/Tool_calling/data/candidates_v2.jsonl` exists, `rl_dataset.py` uses those complete generated trajectories. It assigns records to splits deterministically using the task ID. If the trajectory source is absent, it falls back to the older SFT-derived conversion.

To use a different trajectory file:

```powershell
.\.venv\Scripts\python.exe rl_dataset.py --source path\to\trajectories.jsonl
```

## Train and evaluate

The SFT baseline is trained with:

```powershell
.\.venv\Scripts\python.exe train_baseline_sft.py
```

The current rollout and task-data work prepares the environment for GRPO. A GRPO trainer must consume the prompt rows, run multiple model rollouts through `AirlineEnv`, and return the scorer reward for each completion. Do not use a language-model judge in this loop; the simulator and verifier are deterministic.

Evaluation should compare the base model, SFT adapter, and later GRPO adapter using:

- valid JSON/action rate
- tool-name accuracy
- argument accuracy
- multi-turn tool-sequence accuracy
- policy/confirmation compliance
- final database-state accuracy
- end-to-end task success rate

## Tests

Run the full test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests cover the action parser, current five-tool contract, inference formatting, simulator behavior, scorer behavior, rollout failures, and replayable RL metadata.

