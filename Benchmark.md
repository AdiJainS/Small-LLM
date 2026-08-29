# Diagnostic Benchmark: BFCL

The Berkeley Function Calling Leaderboard evaluates function-calling ability. Unlike τ²-bench, BFCL primarily measures whether a model can correctly invoke functions rather than whether it completes a complete airline workflow.

### BFCL Metrics

Relevant metrics include:

- Overall function-calling accuracy
- Single-turn accuracy
- Multi-turn accuracy
- Multi-step accuracy
- Tool-name accuracy
- Argument accuracy
- AST/function-call validity
- Hallucinated-function detection
- Irrelevant-function detection
- Parallel-function accuracy

### Public Reference Results

The following values are BFCL V4 leaderboard reference points:

```text
Claude Opus 4.5:       77.47% overall accuracy
Qwen3-1.7B native FC:  28.41% overall accuracy
Hammer2.1-1.5B FC:     27.88% overall accuracy
```

The Qwen3-1.7B result is the most relevant public baseline for our model.

### Official BFCL Evaluation

BFCL uses its own official dataset and function definitions. Each example may contain different functions, including single, multiple, parallel, and multi-turn function-calling tasks.

The evaluation pipeline will be:

```text
Official BFCL dataset
    -> BFCL prompt and function definitions
    -> Our Qwen3-1.7B/SFT/GRPO model
    -> BFCL-compatible output
    -> Official BFCL evaluator
    -> Official BFCL metrics
```

This measures general function-calling ability rather than airline database completion.

### BFCL Evaluation Modes

#### Non-Live Evaluation

The evaluator checks the model’s generated function call without executing it.

It evaluates:

- Function name
- Argument names
- Argument values
- Call structure
- AST validity
- Multi-turn call sequences
- Multiple and parallel calls
- Hallucinated functions

No airline database or `AirlineEnv` is required.

#### Live Evaluation

The generated function call is executed by a matching function implementation.

For live evaluation, the implementation must match the BFCL function definition and expected behavior. Our airline environment cannot execute arbitrary BFCL functions outside the airline domain.

We will begin with official BFCL non-live evaluation and only run live evaluation when compatible executors are available.

## Components We Will Reuse

We will reuse the following components from our project:

- Qwen3-1.7B model loading
- SFT checkpoints
- GRPO checkpoints
- LoRA/QLoRA adapters
- Inference code
- Tokenizer configuration
- Batch generation
- Checkpoint selection
- Output logging
- Experiment tracking
- Results storage

These components are model and infrastructure components, so they can be shared across our custom evaluation and official benchmark evaluation.

## Components We Will Not Use as Official BFCL Substitutes

The following components will remain specific to our custom airline benchmark:

- Our four airline tools
- Our airline database
- `AirlineEnv`
- Our airline task shapes
- Our airline-specific reward scorer
- Our local final-state score
- Our custom action parser as a replacement for the BFCL evaluator
- A converted four-tool subset as a replacement for the full BFCL dataset

These components can be used for internal airline evaluation, but they cannot produce an official BFCL score.

## Required Comparison Table

For every external result, we will record:

- Benchmark version
- Evaluator version
- Dataset split
- Model checkpoint
- Parameter count
- Native or prompt-based function calling
- Thinking mode
- User simulator
- Number of trials
- Decoding settings
- Evaluation date

## Success Criteria

The first success criterion is:

```text
Qwen3-1.7B GRPO > Qwen3-1.7B SFT > Qwen3-1.7B Base
```

on our held-out airline benchmark.

The stronger success criterion is improvement on an official external benchmark without increasing:

- Policy violations
- Invalid function calls
- Hallucinated tools
- Incorrect arguments
- Database-state failures

Our final claim should be based on reproducible evaluation using the official benchmark dataset and evaluator.
