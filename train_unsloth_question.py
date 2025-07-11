import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from unsloth import FastLanguageModel
import os

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Get dataset
dataset = load_dataset("./data/medical-o1-reasoning-SFT","zh", split="train")
print(dataset[0])

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>
{}
</think>
{}"""


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for i, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(i, cot, output) + EOS_TOKEN
        texts.append(text)
    return texts

# Load Llama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./model/unsloth/DeepSeek-R1-Distill-Qwen-7B", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    local_files_only = True,
)

## 训练前提问
question = "A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, what would cystometry most likely reveal about her residual volume and detrusor contractions?"

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("before train:"+response[0].split("### Response:")[1])

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    formatting_func=formatting_prompts_func,
    tokenizer = tokenizer,
    args = TrainingArguments(
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 4,
      max_steps= 140,
      warmup_steps = 5,
      learning_rate=5e-4,
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      logging_steps = 10,
      weight_decay=0.01,
      lr_scheduler_type="linear",
      output_dir = "./outputs/unsloth_qwen_sft",
      optim = "adamw_8bit",
      seed = 3407,
  ),
)
trainer.train()

## 训练完之后提问
FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)
print("after train:"+response[0].split("### Response:")[1])


