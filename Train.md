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

### SFT (VERY IMP) 
Accn to a paper Nemotron research N1.

If the paper like arch followed, then - 
Tool definitions + tool-calling tasks
              ↓
        Qwen generates
        reasoning + call
              ↓
        execute/check call
              ↓
         binary reward
              ↓
             GRPO
              ↓
         model improves

You don't have to tell the model exactly what its reasoning should be.
Binary reward system R = 1 if format and tool call is correct , and R =0 otherwise.

under their equal-data-budget setup, 100% RL slightly outperformed 100% Reason-SFT and No-Reason SFT on their tool-calling experiments.

our new arch - 

SFT baseline + Pure-RL experiment .
This arch gave 70+ % accuracy on Qwen2.5 1.5B INstruct . Our qwen model is 3 1.7B , which is better. Hence we might be able to achieve that kind of accuracy.

How our model learns ? If we ask a ques to model , it won't directly tell us the tool which is using for the obv reasons , accn to the paper , it would go through something like 
```
<think>
.....
</think>

<tool_call>
{
  "tool_name": "...",
  "parameters": {
    "X": ".....",
    "Y": "...",
    "Z": ...
  }
}
</tool_call>
```
Correct call is rewarded , wrong call is 0

Reward Structure - 
We do not need a specific reward model 

Qwen generates
       ↓
deterministic verifier
       ↓
reward
       ↓
GRPO
       ↓
Qwen improves

1.R_tool = 1 if correct tool
2.R_args = 1 if arguments correct
3.R_result = 1 if tool result is correctly used
4.R_answer = 1 if final answer passes evaluation
5.R_safety = 1 if safety constraints pass

Both of the reward structures should be implemented and then compared !!

Working of GRPO - 

Suppose GRPO groups as 
`G = {
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    y8
}`
and each of them has some arguments , the reward func will eval each of them 


### Failure mining 
run model on validation - > failures coll. - > label corr traj - > add back to training set - > retrain 

### RL wala part 

