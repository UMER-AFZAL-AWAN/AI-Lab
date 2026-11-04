# =============================================================================
# 20_nlp_and_llms / 02 — LLM FINE-TUNING (SFT)
# =============================================================================
#
# LIKE YOU ARE 5:
# A base LLM (like LLaMA-3) knows a lot but doesn't know how to follow
# instructions. It just completes text: "What is the capital of France? Paris
# is a city in..."  — it rambles rather than answering.
# SFT (Supervised Fine-Tuning) teaches it: "when asked a question, ANSWER it."
# We show it thousands of (question, answer) examples.
#
# C# ANALOGY:
# Base LLM = a library of all C# code on GitHub
# SFT = training it on a dataset of Stack Overflow: (question, accepted answer)
# Now when you ask "how do I sort a List<T>?" it gives a direct, correct answer.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("PART 1: BASE MODEL vs INSTRUCTION-TUNED MODEL")
print("=" * 60)
print("""
BASE MODEL:
  Trained only on next-token prediction on web text.
  Prompt: "The capital of France is"
  Output: "Paris, located in northern France, on the Seine River..."
  → Continues the text, doesn't answer your question directly.

INSTRUCTION-TUNED (SFT) MODEL:
  Prompt: "What is the capital of France?"
  Output: "The capital of France is Paris."
  → Understands the format: question → answer.

SFT training data:
  Thousands of (instruction, response) pairs.
  Sources: human annotators, synthetic data from stronger models, datasets like
    - OpenAssistant (open source)
    - Alpaca (GPT-3.5 generated from 52K instructions)
    - ShareGPT (ChatGPT conversations scraped)
    - FLAN (diverse tasks compiled by Google)
""")

print("=" * 60)
print("PART 2: THE INSTRUCTION FORMAT")
print("=" * 60)
print("""
Each LLM family has its own chat template.
Getting this wrong causes poor performance (model is off-distribution).

LLaMA 3 chat template:
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>
  You are a helpful assistant.
  <|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  What is the capital of France?
  <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
  The capital of France is Paris.
  <|eot_id|>

Mistral / Zephyr template:
  <s>[INST] What is the capital of France? [/INST]
  The capital of France is Paris. </s>

Phi-3:
  <|user|>
  What is the capital of France? <|end|>
  <|assistant|>
  The capital of France is Paris. <|end|>

Using HuggingFace tokenizers:
  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  # → returns correctly formatted string for that model
""")

# Show what the training data looks like (no model needed)
def format_instruction_pair(instruction, response, system_prompt=None):
    """
    Generic format for SFT training.
    In practice, use apply_chat_template from the tokenizer.
    """
    parts = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt}")
    parts.append(f"<|user|>\n{instruction}")
    parts.append(f"<|assistant|>\n{response}")
    return "\n".join(parts)

examples = [
    ("What is backpropagation?",
     "Backpropagation is an algorithm that computes gradients in a neural network "
     "using the chain rule, flowing error signals from the output layer backwards."),
    ("Write a Python function to reverse a string",
     "def reverse_string(s):\n    return s[::-1]"),
    ("Explain dropout in one sentence",
     "Dropout randomly zeroes out neurons during training to prevent the model "
     "from relying on any single neuron, reducing overfitting."),
]

print("SFT training examples:")
for inst, resp in examples[:2]:
    formatted = format_instruction_pair(inst, resp)
    print(f"\n{'-'*50}")
    print(formatted)
print(f"\n{'-'*50}")
print(f"(+ {len(examples)-2} more examples...)")

print()
print("=" * 60)
print("PART 3: WHAT GETS TRAINED — THE LOSS MASK")
print("=" * 60)
print("""
Key insight: we ONLY compute loss on the assistant's tokens.
The user's question tokens are part of the input but NOT the target.

Example tokens:
  <|user|> What is ... <|assistant|> Paris is ...
                        ↑ loss starts here

If we computed loss on user tokens too:
  - Model would try to predict "What is..." after seeing nothing
  - That's "unlearnable" — hurts training

Implementation:
  1. Tokenise the full conversation
  2. Find the position of <|assistant|>
  3. Set labels = -100 for all tokens BEFORE assistant  ← CrossEntropyLoss ignores -100
  4. labels[assistant_start:] = input_ids[assistant_start:]  ← train on these
""")

# Visualise the loss mask
tokens   = ["<user>", "What", "is", "<asst>", "Paris", "is", "great", "<eos>"]
input_ids = list(range(len(tokens)))
labels    = [-100] * 4 + input_ids[4:]   # -100 for user tokens

print("Token sequence with loss mask:")
for tok, iid, lab in zip(tokens, input_ids, labels):
    ignored = " (ignored)" if lab == -100 else " ← train on this"
    print(f"  {tok:12s}  id={iid}  label={lab:5d}{ignored}")

print()
print("=" * 60)
print("PART 4: SFT WITH HUGGINGFACE (practical code)")
print("=" * 60)
print("""
Using TRL (Transformer Reinforcement Learning) library for SFT:
  pip install trl transformers datasets

  from transformers import AutoModelForCausalLM, AutoTokenizer
  from trl import SFTTrainer, SFTConfig

  # Load model (small model for demonstration)
  model_name = "microsoft/phi-2"  # 2.7B params, fits in 8GB VRAM with LoRA

  model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # Dataset: each row has {"messages": [...]} or {"text": "..."}
  from datasets import load_dataset
  dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

  # SFT Trainer handles chat templating, loss masking, etc.
  training_args = SFTConfig(
      output_dir="./sft_output",
      max_steps=1000,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,  # effective batch = 16
      learning_rate=2e-5,
      bf16=True,
      max_seq_length=1024,
      dataset_kwargs={"skip_prepare_dataset": True},
  )

  trainer = SFTTrainer(
      model=model,
      tokenizer=tokenizer,
      train_dataset=dataset,
      args=training_args,
  )
  trainer.train()
""")

print("=" * 60)
print("PART 5: SFT + LoRA (fine-tuning on 8GB VRAM)")
print("=" * 60)
print("""
Full SFT of a 7B model requires ~80GB VRAM.
With LoRA + 4-bit quantisation, 7B SFT fits in 8GB.

  from peft import get_peft_model, LoraConfig, TaskType
  from transformers import BitsAndBytesConfig

  # 4-bit quantisation (NF4 reduces model from 14GB to 4GB)
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
  )

  # Apply LoRA (only train a small fraction of weights)
  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16,                        # rank (16-64 common for SFT)
      lora_alpha=32,               # scaling = alpha/r = 2.0
      target_modules=["q_proj", "v_proj"],  # LLaMA attention projections
      lora_dropout=0.05,
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  # → "trainable params: 4,194,304 || all params: 6,742,614,784 || 0.062%"

VRAM breakdown for 7B LLaMA + QLoRA:
  Base model (4-bit): ~4GB
  Gradients (LoRA):   ~0.5GB
  Activations:        ~2GB
  Total:              ~6.5GB  ← fits in 8GB RTX 5060!
""")

print("=" * 60)
print("PART 6: SFT vs RLHF vs DPO")
print("=" * 60)
print("""
SFT (Supervised Fine-Tuning):
  Train on (instruction, gold response) pairs.
  Simple, fast, effective.
  Limitation: requires high-quality labelled data.
  → Creates: Alpaca, Vicuna, early ChatGPT

RLHF (Reinforcement Learning from Human Feedback):
  1. Train a reward model: given two responses, which is better? (human labels)
  2. Use PPO (a RL algorithm) to optimise the LLM to maximise reward.
  Pros: aligns model to human preferences.
  Cons: complex, unstable training, expensive reward model.
  → Used by: InstructGPT (ChatGPT), LLaMA-2 Chat

DPO (Direct Preference Optimisation):
  Simpler alternative to RLHF. Same preference data (chosen/rejected pairs)
  but optimised directly without a separate reward model.
  Formula: maximise log P(chosen) − log P(rejected)
  Pros: simpler than RLHF, comparable quality.
  → Used by: Zephyr, most recent open models.

Typical training pipeline for a modern LLM:
  1. Pre-training (next token on web)       ← base model
  2. SFT (instruction following)            ← instruction model
  3. DPO or RLHF (align to preferences)     ← chat model
""")

# =============================================================================
# EXPECTED OUTPUT:
# Formatted instruction examples
# Loss mask: user tokens have label=-100, assistant tokens have positive labels
# =============================================================================
