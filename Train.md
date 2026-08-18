# Training Arch
Training Ideas - 

### ToolFormer - 

Tool call insertion ( Teaches model to when to call tools and argumetation passing )

Filtering ( Keeping safe tool trajectories)

### ReAct - 

Qwen3 1.7B Instruct - > reasoning - > call tool - > observe - > next step - > call other tool

### ToolLLM -

RapidAPI multi-tool hierarchy info.

Can construct a decision tree to expand search space and increase possibility of finding a valid path

--------------------------------------------------------------------------------

Raw domain data / public datasets / APIs

1. Tool Schema Bank

2. Synthetic Instruction Generation

3. Tool-Use Trajectory Generation

4. Tool Execution + Result Collection

5. Filtering / Verification

6. SFT on Qwen3-1.7B (maybe)

7. Failure Mining

8. Preference Dataset

9. DPO / Optional RL

10. Final Evaluation

-----------------------------------------------------------------------------------

### Tool Use Trajector Gen - 
User ques - > tool call - > tool output - >next tool  - > ans

### Domain grounding
RAFT style  - AFT’s useful idea: train the model in an “open-book” setting with relevant documents and distractors, so it learns to use correct evidence and ignore noise.

### SFT 
We may even not require it accn to a paper

### Failure mining 
run model on validation - > failures coll. - > label corr traj - > add back to training set - > retrain 

### RL wala part 

