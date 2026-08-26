Step 1 - Build the action parser. Accept TOOL_CALL, FINAL_ANSWER, and ASK_USER; reject
malformed or mixed outputs.

Step 2 - Connect the airline tool executor. Return actual structured observations and structured
errors.

Step 3 - Implement the multi-turn rollout loop. Generate one action, execute one tool, append its
observation, and continue.

Step 4 - Implement the deterministic verifier. Check tool identity, all parameter values, confirmation,
policy, facts, and final reservation state.

Step 5 - Write ten hand-checked airline tasks: lookup, direct-flight search, one-stop search, baggage
update, passenger update, cancellation, confirmation refusal, error recovery, policy answer, and
no-tool answer.

Step 6 - Run the untrained Qwen3-1.7B baseline. Record complete success, argument accuracy,
no-tool accuracy, policy pass rate, recovery rate, and calls per task.

Step 7 - Run a tiny GRPO smoke test on one read-only tool and one write tool before adding every
airline tool.

Step 8 - Add multi-step reservation tasks: identify user -> inspect reservation -> search or update ->
verify result -> final response.

Step 9 - Run binary versus decomposed reward ablations.

Step 10 - Add Fission-style recovery examples from on-policy errors.

Step 11 - Scale the best 1.7B recipe to Qwen3-4B.
