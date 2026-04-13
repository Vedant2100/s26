In this tutorial, you will distinguish instruction data from preference pairs and conversation data, implement chat templates with correct loss masking, estimate data collection costs for each format, and detect common quality issues such as position bias and length bias in annotation data.

## Prerequisites: Tokenization Basics[#](#prerequisites-tokenization-basics)

▶What is tokenization? (Refresher)

Tokenization is the process of breaking text into discrete units (tokens) that the model processes. This matters for data formats because:

**Token types:**

* **Words/Subwords:** "hello" → ["hello"] or ["hel", "lo"]
* **Special tokens:** `<|start_header_id|>`, `<|end_header_id|>`, `[BOS]`, `[EOS]`, etc.
* **Padding/Masking:** Some tokens are marked "don't learn from this"

**Why it matters for fine-tuning:**

When you have a conversation like:

```
User: What is 2+2?
Assistant: 4
```

The tokenizer needs to know: "Learn from the assistant's response, but NOT from 'User: What is 2+2?'." This is done via **loss masks** -- binary arrays indicating which tokens contribute to the training objective.

**Key insight:** A poorly designed chat template might accidentally:

* Train the model on the *prompt* (bad -- wastes compute)
* Have misaligned token boundaries (confusing the model)
* Use inconsistent special tokens (creates distribution shift)

▶What is a loss mask? (Refresher)

A loss mask is a binary sequence indicating which tokens contribute to the training loss.

Example:

```
Text:     "User: 2+2?  [SEP]  Assistant: 4  [EOS]"
Tokens:   [1, 2, 3, 4,  5,      6, 7, 8, 9, 10]
Mask:     [0, 0, 0, 0,  0,      1, 1, 1, 1, 1]
```

Only tokens where mask=1 contribute to the loss. This ensures the model learns:

* The format structure (User: ... Assistant: ...)
* How to respond to prompts (Assistant: tokens)
* But NOT the user's prompt itself (redundant, wastes compute)

In SFT, you typically:

* Mask out the user prompt (loss\_mask=0)
* Keep only the assistant response (loss\_mask=1)

In preference learning (RLHF/DPO), you:

* Mask out the prompt (same as both chosen and rejected)
* Keep only the response tokens

This ensures *preference learning* focuses on differences in responses, not prompt encoding.

## Instruction Data (for SFT)[#](#instruction-data-for-sft)

The simplest format: pairs of (instruction, response).

## Instruction Data (for SFT)[#](#instruction-data-for-sft)

The simplest format: pairs of (instruction, response).

instruction\_format.pycpu-only

```
import json

# Instruction data format
instruction_examples = [
  {
      "instruction": "What is the capital of France?",
      "response": "The capital of France is Paris."
  },
  {
      "instruction": "Write a Python function to calculate factorial.",
      "response": '''def factorial(n):
  """Calculate factorial of n."""
  if n <= 1:
      return 1
  return n * factorial(n - 1)'''
  },
  {
      "instruction": "Explain why the sky is blue in simple terms.",
      "response": "The sky appears blue because of a phenomenon called Rayleigh scattering. Sunlight contains all colors, but blue light has a shorter wavelength and gets scattered more by tiny molecules in the atmosphere. This scattered blue light reaches our eyes from all directions, making the sky look blue."
  },
  {
      "instruction": "Summarize this text: [The Industrial Revolution began in Britain...]",
      "input": "The Industrial Revolution began in Britain in the late 18th century and transformed society from agrarian to industrial. Key innovations included the steam engine, spinning jenny, and power loom.",
      "response": "The Industrial Revolution started in late 18th century Britain, shifting society from farming to manufacturing through innovations like the steam engine and textile machinery."
  }
]

print("Instruction Data Format Examples")
print("=" * 60)
```

## Chat Templates: Encoding Conversations[#](#chat-templates-encoding-conversations)

A **chat template** is a standardized way to format multi-turn conversations into a single sequence of tokens. Different frameworks use different templates. This is critical because:

1. **The tokenizer must align message boundaries** with special tokens
2. **Loss masks must be applied consistently** across turns
3. **The model learns the structure** through repeated exposure

### Common Chat Template Formats[#](#common-chat-template-formats)

Multi-turn Conversation  
(User, Assistant, System)

ChatML  
OpenAI, vLLM

Llama  
Meta format

Alpaca  
Stanford format

Custom  
Your own

<|im\_start|>role  
content  
<|im\_end|>

<<SYS>>...<</SYS>>  
[INST] ... [/INST]

### Instruction:  
### Response:

Your markers  
Your structure

### ChatML (OpenAI / vLLM Standard)[#](#chatml-openai-vllm-standard)

ChatML is the most common modern format:

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
The answer is 4.
<|im_end|>
```

**Token breakdown:**

* `<|im_start|>` -- Special token marking message boundary (learned token)
* `system/user/assistant` -- Role identifier
* Message content
* `<|im_end|>` -- Special token marking message end
* Loss mask: `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ...]` (only assistant response counted)

### Llama Format[#](#llama-format)

Meta's Llama uses a different style:

```
<s>[INST] <<SYS>>
You are helpful.
<</SYS>>

What is 2+2? [/INST] The answer is 4. </s>
```

**Key differences:**

* System prompt nested inside `<<SYS>>` tags
* Instructions wrapped in `[INST]...[/INST]`
* Multiple turns handled by repeating instruction blocks
* More compact than ChatML

chat\_templates.pycpu-only

```
def apply_chatml_template(messages: list) -> str:
  """
  Convert list of messages to ChatML format.
  messages = [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
  ]
  """
  formatted = ""
  for message in messages:
      role = message["role"]
      content = message["content"]
      formatted += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
  return formatted.rstrip("\n")

def apply_llama_template(messages: list) -> str:
  """
  Convert to Llama format.
  Assumes alternating user/assistant turns with optional system.
  """
  formatted = "<s>"

  # Find and prepend system message
  system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

  # Process user/assistant pairs
  user_assistant_pairs = [m for m in messages if m["role"] in ["user", "assistant"]]

  for i in range(0, len(user_assistant_pairs), 2):
```

### Special Tokens in Chat Templates[#](#special-tokens-in-chat-templates)

Different models use different special tokens:

▶Common special tokens and their purposes

| Token | Format | Purpose | Loss Mask |
| --- | --- | --- | --- |
| `<|im_start|>` | ChatML | Mark message start | Usually 0 |
| `<|im_end|>` | ChatML | Mark message end | Usually 1 |
| `[INST]` | Llama | Start of instruction | 0 |
| `[/INST]` | Llama | End of instruction | 0 |
| `<<SYS>>` | Llama | System message marker | 0 |
| `<s>` or `[BOS]` | Both | Beginning of sequence | 0 |
| `</s>` or `[EOS]` | Both | End of sequence | Usually 1 |
| `<pad>` | Both | Padding token | 0 |

**Critical insight:** The loss mask pattern differs by role:

* **User/System tokens:** mask = 0 (don't learn from these)
* **Assistant tokens:** mask = 1 (learn from these)
* **Special tokens themselves:** Often mask = 0 (they're structural)

This ensures the model:

1. Learns to *predict* responses, not memorize prompts
2. Understands conversation structure through repeated exposure
3. Doesn't waste gradient computation on already-known inputs

## Preference Data (for RLHF/DPO)[#](#preference-data-for-rlhfdpo)

Triples of (prompt, chosen, rejected) teach relative quality.

preference\_format.pycpu-only

```
# Preference data format
preference_examples = [
  {
      "prompt": "How do I improve my writing?",
      "chosen": "Here are some effective ways to improve your writing:\n\n1. **Read widely** - Exposure to different styles builds intuition\n2. **Write daily** - Consistent practice is key\n3. **Get feedback** - Fresh eyes catch what you miss\n4. **Edit ruthlessly** - Good writing is rewriting\n5. **Study grammar** - Rules give you tools to break them intentionally",
      "rejected": "just write more lol"
  },
  {
      "prompt": "What's 15% of 80?",
      "chosen": "15% of 80 is 12.\n\nCalculation: 0.15 × 80 = 12\n\nAlternatively: 10% of 80 is 8, and 5% is 4, so 15% = 8 + 4 = 12.",
      "rejected": "15% of 80 is 14."  # Wrong answer
  },
  {
      "prompt": "Tell me how to hack into my ex's email",
      "chosen": "I can't help with accessing someone else's email account without authorization. This would be illegal in most jurisdictions and a violation of their privacy.\n\nIf you're having trouble moving on after a breakup, I'd be happy to suggest some healthier coping strategies.",
      "rejected": "Here's how you could try to access their email: First, try common passwords they might use..."
  },
]

print("Preference Data Format Examples")
print("=" * 60)

for i, example in enumerate(preference_examples, 1):
  print("\nExample %d:" % i)
  print("  Prompt:   %s..." % example['prompt'][:50])
  print("  Chosen:   %s..." % example['chosen'][:50])
  print("  Rejected: %s..." % example['rejected'][:50])

print("\n" + "=" * 60)
print("What preferences capture:")
```

## Conversation Data (for Chat Models)[#](#conversation-data-for-chat-models)

Multi-turn dialogues with role markers.

conversation\_format.pycpu-only

```
# Conversation format (ChatML-style)
conversation_examples = [
  {
      "messages": [
          {"role": "system", "content": "You are a helpful coding assistant."},
          {"role": "user", "content": "How do I read a file in Python?"},
          {"role": "assistant", "content": "You can read a file using the open() function with a context manager:\n\n```python\nwith open('filename.txt', 'r') as f:\n    content = f.read()\n```"},
          {"role": "user", "content": "What if the file doesn't exist?"},
          {"role": "assistant", "content": "You can handle that with a try-except block:\n\n```python\ntry:\n    with open('filename.txt', 'r') as f:\n        content = f.read()\nexcept FileNotFoundError:\n    print('File not found')\n```"}
      ]
  }
]

print("Conversation Format (Multi-turn)")
print("=" * 60)

for conv in conversation_examples:
  for msg in conv["messages"]:
      role = msg["role"].upper()
      content = msg["content"][:60].replace("\n", " ")
      print("  [%-10s] %s..." % (role, content))

print("\n" + "=" * 60)
print("Key features:")
print("  - System prompts set persona/behavior")
print("  - Multi-turn maintains context")
print("  - Each assistant turn is a training target")
print("  - Can convert to instruction format for SFT")

# Important: loss masking in multi-turn
```

## Break It: Common Chat Template Mistakes[#](#break-it-common-chat-template-mistakes)

Let's see what happens when you mess up your chat template or loss masking:

break\_chat\_templates.pycpu-only

```
def demonstrate_template_errors():
  """Show what goes wrong with bad chat formatting."""

  print("MISTAKE #1: Inconsistent Role Markers")
  print("=" * 60)
  bad_template = """user: What is 2+2?
assistant: The answer is 4.
User: What about 3+5?
assistant: That would be 8."""

  good_template = """<|im_start|>user
What is 2+2?
<|im_end|>
<|im_start|>assistant
The answer is 4.
<|im_end|>
<|im_start|>user
What about 3+5?
<|im_end|>
<|im_start|>assistant
That would be 8.
<|im_end|>"""

  print("BAD (inconsistent capitalization, no special tokens):")
  print(bad_template)
  print()
  print("GOOD (consistent structure, clear boundaries):")
  print(good_template)
  print()
  print("Impact: Bad formatting → model learns to output")
```

## Data Collection Methods[#](#data-collection-methods)

Quality Control

Data Sources

Human Annotation  
Gold standard, expensive

Synthetic Generation  
Cheap, needs curation

Distillation  
From stronger model

Red Teaming  
Adversarial examples

Human Review

Automated Filtering

Deduplication

Final Dataset

data\_collection.pycpu-only

```
# Data collection cost analysis
def estimate_collection_cost(
  num_examples: int,
  method: str,
  format_type: str = "instruction"
) -> dict:
  """
  Estimate cost and time for data collection.
  """

  # Cost per example (rough estimates)
  costs = {
      "human_annotation": {
          "instruction": 2.00,  # $/example
          "preference": 5.00,   # Comparing requires more thought
          "conversation": 10.00,  # Multi-turn is expensive
          "time_per_example": 5,  # minutes
      },
      "synthetic": {
          "instruction": 0.01,
          "preference": 0.05,
          "conversation": 0.10,
          "time_per_example": 0.01,
      },
      "distillation": {
          "instruction": 0.02,
          "preference": 0.10,
          "conversation": 0.20,
          "time_per_example": 0.02,
      },
```

## Data Quality Dimensions[#](#data-quality-dimensions)

data\_quality.pycpu-only

```
import re
from typing import List, Dict

def analyze_data_quality(examples: List[Dict]) -> Dict:
  """
  Analyze quality dimensions of instruction data.
  """
  issues = {
      "too_short": [],
      "too_long": [],
      "missing_instruction": [],
      "empty_response": [],
      "potential_duplicates": [],
      "formatting_issues": [],
  }

  seen_instructions = set()

  for i, ex in enumerate(examples):
      instruction = ex.get("instruction", "")
      response = ex.get("response", "")

      # Check for missing fields
      if not instruction:
          issues["missing_instruction"].append(i)
      if not response:
          issues["empty_response"].append(i)

      # Check lengths
      if len(response) < 10:
```

## Conversation Structure: Multi-turn Dynamics[#](#conversation-structure-multi-turn-dynamics)

When training on multi-turn conversations, there's a subtle but important question: **how many turns should each example contain?**

Multi-turn  
Conversation Strategy

Few-turn  
1-2 exchanges

Medium-turn  
3-5 exchanges

Long-turn  
5+ exchanges

Pros: Fast, clear signal

Cons: Limited context

Pros: Balanced learning

Cons: Moderate length

Pros: Rich context, memory

Cons: Long sequences, context dilution

**Key observations:**

* **Few-turn (1-3 exchanges):** Each conversation is short and focused. Model learns clear stimulus-response patterns. Good for instruction following.
* **Medium-turn (3-7 exchanges):** Realistic conversations. Model learns to maintain context and adapt to follow-ups. Closer to real usage.
* **Long-turn (7+ exchanges):** Tests the model's ability to remember information over time. Can expose failures in attention/context.

**Training signal implication:**

Longer conversations provide more training examples per file (e.g., a 5-turn conversation gives you 3-5 assistant responses to train on), but each response is conditioned on more history--potentially noisier if early context is forgotten.

## Data Requirements by Stage[#](#data-requirements-by-stage)

| Stage | Typical Size | Key Quality Factors |
| --- | --- | --- |
| **SFT** | 10K-100K examples | Diversity, correctness, format consistency |
| **Reward Model** | 50K-500K comparisons | Annotator agreement, calibration |
| **RLHF** | Generated on-the-fly | Reward model quality determines ceiling |
| **DPO** | 10K-100K preferences | Preference margin clarity |

## Annotation Guidelines[#](#annotation-guidelines)

annotation\_guidelines.pycpu-only

```
# Example annotation guidelines for preference data
annotation_guidelines = """
PREFERENCE ANNOTATION GUIDELINES
================================

Task: Given a prompt and two responses (A and B), select the better response.

CRITERIA (in order of importance):
1. CORRECTNESS: Factually accurate > Contains errors
2. HELPFULNESS: Addresses the user's actual need
3. SAFETY: Refuses harmful requests appropriately
4. CLARITY: Well-organized, easy to understand
5. CONCISENESS: Appropriate length (not too verbose or too brief)

DECISION RULES:
- If both equally correct, prefer more helpful
- If both equally helpful, prefer safer
- If both equally safe, prefer clearer
- When genuinely equal, mark as "tie"

EDGE CASES:
- Harmful requests: Prefer response that refuses appropriately
- Ambiguous questions: Prefer response that seeks clarification
- Creative tasks: Prefer more engaging while maintaining quality

QUALITY METRICS:
- Take 60-90 seconds per comparison
- If unsure after 2 minutes, mark as "uncertain"
- Aim for >80% agreement with gold labels in calibration
```

## Format-Specific Data Quality Issues[#](#format-specific-data-quality-issues)

Before general quality checks, make sure your **format is consistent**:

## Common Data Quality Issues[#](#common-data-quality-issues)

bias\_detection.pycpu-only

```
import numpy as np

def detect_position_bias(preferences: list) -> dict:
  """
  Detect if annotators prefer first or second option.
  """
  first_chosen = sum(1 for p in preferences if p == "A")
  second_chosen = sum(1 for p in preferences if p == "B")
  total = first_chosen + second_chosen

  position_bias = abs(first_chosen - second_chosen) / total

  return {
      "first_chosen_rate": first_chosen / total,
      "second_chosen_rate": second_chosen / total,
      "position_bias_score": position_bias,
      "is_biased": position_bias > 0.1  # >10% difference is concerning
  }

def detect_length_bias(data: list) -> dict:
  """
  Detect if longer responses are systematically preferred.
  """
  longer_chosen = 0
  shorter_chosen = 0

  for item in data:
      chosen_len = len(item.get("chosen", ""))
      rejected_len = len(item.get("rejected", ""))
```

## Complete Example: From Raw Data to Training Format[#](#complete-example-from-raw-data-to-training-format)

Let's see a realistic pipeline: converting raw conversations to a properly formatted, masked dataset:

complete\_pipeline.pycpu-only

```
import json

# Step 1: Raw conversation data (as it comes from logs)
raw_conversations = [
  {
      "id": "conv_001",
      "messages": [
          {"role": "user", "content": "What is 2+2?", "timestamp": "2024-01-01T10:00:00"},
          {"role": "assistant", "content": "The answer is 4. This is basic arithmetic: 2 + 2 = 4.", "timestamp": "2024-01-01T10:00:05"},
          {"role": "user", "content": "What about 2+3?", "timestamp": "2024-01-01T10:00:10"},
          {"role": "assistant", "content": "That would be 5: 2 + 3 = 5.", "timestamp": "2024-01-01T10:00:15"},
      ]
  }
]

# Step 2: Convert to ChatML format with loss masks
def convert_to_chatml_with_masks(conversation: dict) -> dict:
  """
  Convert raw conversation to ChatML with computed loss masks.

  Note: In production, you would use the model's actual tokenizer
  (e.g., tiktoken, sentencepiece) instead of character-level masks.
  This uses character-level masking for demonstration purposes.
  """
  formatted_text = ""
  loss_mask = []

  for msg in conversation["messages"]:
      role = msg["role"]
      content = msg["content"]
```

## Scale Thought Experiment[#](#scale-thought-experiment)

| Scale | Data Challenge | Mitigation |
| --- | --- | --- |
| **1K examples** | Every example matters | Expert curation, multiple reviews |
| **10K examples** | Consistency across annotators | Guidelines, calibration sessions |
| **100K examples** | Annotation cost prohibitive | Synthetic generation + filtering |
| **1M+ examples** | Manual review impossible | AI filtering, automatic quality metrics |

## Practical Decision: Choosing Your Format[#](#practical-decision-choosing-your-format)

When you're building a fine-tuning pipeline, you need to decide on one canonical format. Here's a decision tree:

▶Which format should I use?

**Use instruction format if:**

* You have single-turn Q&A pairs (most common)
* You're doing supervised fine-tuning (SFT) only
* You want simplicity and fast iteration
* Examples: LIMA, Alpaca, most public datasets

**Use conversation format if:**

* You have multi-turn dialogues
* You want to teach the model dialogue coherence
* You're planning RLHF/DPO (preferences operate on conversations)
* Examples: OpenAI's InstructGPT, most chat models

**Use custom format if:**

* You have domain-specific structure (medical charts, code diffs, etc.)
* You need special handling of specific fields
* You're optimizing for your specific use case
* Make sure you have strong reasons and clear documentation

**General advice:**

1. Start with ChatML (most standardized)
2. Make sure your tokenizer has the special tokens
3. Test loss masking on a small sample manually
4. Use the same format throughout your dataset
5. Document your format choices (future you will thank you)

format\_decision\_helper.pycpu-only

```
def choose_format(
  num_turns_avg: float,
  has_system_prompts: bool,
  doing_preference_learning: bool,
  simplicity_priority: bool
) -> str:
  """
  Simple heuristic for choosing a chat format.
  """

  if num_turns_avg <= 1.5 and not has_system_prompts and simplicity_priority:
      return "instruction (simple)"

  if num_turns_avg > 1.5 or has_system_prompts or doing_preference_learning:
      return "ChatML (recommended)"

  if num_turns_avg > 5:
      return "ChatML + careful context handling"

  return "ChatML (default choice)"

# Test the helper
scenarios = [
  (1.0, False, False, True, "Simple Q&A dataset, SFT only"),
  (3.0, True, False, False, "Multi-turn with system prompts, SFT"),
  (2.5, True, True, False, "Multi-turn, planning RLHF"),
  (0.9, False, True, True, "Preference learning on single-turn"),
]

print("Format Selection Guide")
```

## Production Reality[#](#production-reality)

**OpenAI's approach (InstructGPT):**

* 40 contractors for annotation
* Detailed guidelines with examples
* Regular calibration sessions
* Separate teams for SFT vs preference data

**Anthropic's approach (Constitutional AI):**

* Use AI to generate preference data
* Principles guide the AI's judgments
* Scale with compute, not human labor
* Human oversight for principle design

## Checkpoint Questions[#](#checkpoint-questions)

Use these to verify understanding before moving on:

1. A ChatML-formatted conversation has 3 user turns (average 30 tokens each) and 3 assistant turns (average 120 tokens each). Calculate the total sequence length and the percentage of tokens that contribute to the SFT loss (only assistant tokens are unmasked).
2. You need 10K preference pairs for DPO training. Estimate the annotation cost if human annotators take 90 seconds per comparison at 25 USD/hour. Compare this to generating the same pairs synthetically at 0.05 USD per pair via an API.
3. Your preference dataset shows that the chosen response is longer than the rejected response in 78% of cases. Is this evidence of length bias? What is the bias score (scale 0 to 1), and what concrete step would you take to mitigate it?