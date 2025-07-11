import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification,AutoModelForSequenceClassification

model_name = "./model/qnli-electra-base"  # 选择一个文本分类模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
print(round(loss.item(), 2))
