1. Read the dataset first , extract rows/cols/tools from dataset.
2. Decide chat format
3. Training format
4. 
   {
  "action": "tool_call",
  "tool_name": "lookup_order",
  "arguments": {
    "order_id": "12345"
  }
}
For final answer:
{
  "action": "final_answer",
  "content": "Your order has been cancelled and the refund has been initiated."
}

5.Initially convert 500 examples.
6.Tiny SFT smoke test - just to check correct working of data format tokenizers etc..

As later on RL can be tougher to debug , we may not even know the issues GRPO faced .
(Baseline trn)

7.Tau bench eval 

8.Full SFT On APIGen-MT-5k

9.Eval

10.Failure mining

11.Make corrected examples from failures and retrain

12.GRPO

13.Fin


