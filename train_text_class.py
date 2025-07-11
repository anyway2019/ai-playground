import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from unsloth import FastLanguageModel
import os

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# load dataset
dataset = load_dataset("./data/ag_news","top4-balanced",split="complete")
# split dataset to train and test
train_test_split = dataset.train_test_split(test_size=0.2, train_size=0.8,shuffle=True,seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.


### Instruction:
You are a expert with advanced knowledge in text classification. 
Please classify for a given input text with one label in below label list? 

### label:
- Business
- Entertainment
- Europe
- Health
- Italia
- Music Feeds
- Sci/Tech
- Software and Developement
- Sports
- Toons
- Top News
- Top Stories
- U.S.
- World

### input:
Text: {}

### output:
<answer>The answer is: {}"""


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.


### Instruction:
You are a expert with advanced knowledge in text classification. 
Please classify for a given input text with one label in below label list? 
- Business
- Entertainment
- Europe
- Health
- Italia
- Music Feeds
- Sci/Tech
- Software and Developement
- Sports
- Toons
- Top News
- Top Stories
- U.S.
- World

### input:
Text: {}

### output:
<answer>The answer is: {}"""


question = "By MIKE SCHNEIDER CAPE CANAVERAL, Fla. - The crew of space shuttle Atlantis took scrupulous notes during two of last year's construction missions at the international space station, because they will be expected to perform some of the same tasks."

def formatting_prompts_func(examples):
    inputs = examples["text"]
    outputs = examples["category"]
    texts = []
    for i, output in zip(inputs, outputs):
        text = train_prompt_style.format(i, output) + EOS_TOKEN
        texts.append(text)
    return texts

# load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./model/unsloth/DeepSeek-R1-Distill-Qwen-7B", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    local_files_only = True,
)

## inference before train
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("before train:"+response[0].split("output:")[1])

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
    train_dataset = train_dataset,
    eval_dataset= eval_dataset,
    formatting_func=formatting_prompts_func,
    tokenizer = tokenizer,
    args = TrainingArguments(
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 4,
      max_steps = 200,
      warmup_steps = 10,
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

## inference after train
FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs)
print("after train:"+response[0].split("output:")[1])

