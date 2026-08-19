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

-----------------------------------------------------------------------------------
Training Arch - 

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
### Instruction generation
Create a scenario/strucutured query and then turn it into a natural lang

ground-truth scenario object independently of the text must be maintained

Intent taxonomy - ie what model is doing to acheiev its goals

### Tool Use Trajector Gen - 
We will use ReAct style answering pattern.

User ques - > model decision - >  tool call - > tool output - > model decision - > next tool  - > ans

Our model receives a current state defined as S_t .
`
S_t = user ques + prev dec + prev tool calls + prev res + info
`
and based upon that an action state defined as A_t is being generated 
` A_t ∈ {
    TOOL_CALL,
    FINAL_ANSWER,
    ASK_USER
}
`
Hence  ` A_t = π(S_t)  where π is the model's learned policy.` 
The reason for multiple model decision is to not make our model to blindly follow 1st model dec. The 2nd model decision sees query + tool call + tool state and now checks if the tool is suffice and give the final answer.

                 MODEL DECISION
                       │
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Do I need       Which tool?      What arguments?
    a tool?
       │               │                │
       └───────────────┼────────────────┘
                       ↓
                 Is another step
                   necessary?
                       ↓
                Final answer?

### Tool Execution + Result Collection
Never have the teacher invent the tool result as it can make up things.

Initially we will use deterministic tools - same output whenever it receives the same input, assuming its underlying rules/data don't change. Imagine we generate many traj. , laer we can generate similiar traj. as tool is deterministic  which helps in reproductibility ,debug,stable eval .

Static data (in the form of csvs) can also be used.

### Filtration
Quality control on the basis of 

1. Schema Validity
2. Tool Existence
3. Argumenting correctness
4. Tool necessity
5. Result consistency
6. Safety verification
On the basis of these , we can give scores , maybe derive a formal for scores

### Domain ground
RAFT style  - AFT’s useful idea: train the model in an “open-book” setting with relevant documents and distractors, so it learns to use correct evidence and ignore noise.

### SFT 
We may even not require it accn to a paper

### Failure mining 
run model on validation - > failures coll. - > label corr traj - > add back to training set - > retrain 

### RL wala part 

