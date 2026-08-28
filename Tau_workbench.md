1. Action parser and inference updation
2. Rollout loop implementation - 
parse_action -> valid action (if no then failed rollout and reward) -> tool call ->AirlineEnv ->Append structured -> generate model action 
Model format -

```
User: "Cancel reservation R00012."

Model output:
{"action":"tool_call","tool_name":"cancel_reservation",
 "arguments":{"reservation_id":"R00012"}}

Parser: valid
Environment: executes cancel_reservation(...)
Tool observation:
{"ok": true, "reservation_id": "R00012", "status": "cancelled"}

Model sees that observation and produces:
{"action":"final_answer","content":"Your reservation R00012 has been cancelled."}

Verifier: checks the tool, arguments, policy, and final database state.
Reward: high if all checks pass.
```
