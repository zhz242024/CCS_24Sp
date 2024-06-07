import pickle

with open('feature/train_label.pickle', 'rb') as f:
    train_labels = pickle.load(f)

with open('feature/dev_label.pickle', 'rb') as f:
    dev_labels = pickle.load(f)

def convert_phq8_score_to_label(score):
    if 0 <= score <= 4:
        return 0
    elif 5 <= score <= 9:
        return 1
    elif 10 <= score <= 14:
        return 2
    elif 15 <= score <= 19:
        return 3
    elif 20 <= score <= 24:
        return 4
    else:
        raise ValueError("Invalid PHQ-8 score")

import os
from datasets import Dataset, load_metric, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import random

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

data = {
    "file_path": [],
    "binary_label": [],
    "score_label": [],
    "gender": []
}

dev_data = {
    "file_path": [],
    "binary_label": [],
    "score_label": [],
    "gender": []
}

selected_keys = random.sample(list(train_labels.keys()), 5)
train_labels = {key: train_labels[key] for key in selected_keys}

for participant_id, label_info in train_labels.items():
    file_path = os.path.join('data/train_uttr', f"spk_{participant_id}_uttr0.wav")
    idx = 1
    while os.path.isfile(file_path):
        if os.path.exists(file_path):
            binary_label, score_label, gender = label_info
            score_label = convert_phq8_score_to_label(score_label)
            data["file_path"].append(file_path)
            data["binary_label"].append(binary_label)
            data["score_label"].append(score_label)
            data["gender"].append(gender)
        file_path = os.path.join('data/train_uttr', f"spk_{participant_id}_uttr{idx}.wav")
        idx += 1

dataset = Dataset.from_dict(data)

for participant_id, label_info in dev_labels.items():
    file_path = os.path.join('data/dev_uttr', f"spk_{participant_id}_uttr0.wav")
    idx = 1
    while os.path.isfile(file_path):
        if os.path.exists(file_path):
            binary_label, score_label, gender = label_info
            score_label = convert_phq8_score_to_label(score_label)
            dev_data["file_path"].append(file_path)
            dev_data["binary_label"].append(binary_label)
            dev_data["score_label"].append(score_label)
            dev_data["gender"].append(gender)
        file_path = os.path.join('data/dev_uttr', f"spk_{participant_id}_uttr{idx}.wav")
        idx += 1
        

dev_dataset = Dataset.from_dict(dev_data)

import librosa
import soundfile

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", attn_implementation="flash_attention_2")

def process_data_2(data):
    speech_list = []
    for file_path in data["file_path"]:
        audio, _ = librosa.load(file_path, sr=16000)
        speech_list.append(audio)
    result = processor(speech_list, sampling_rate=16000, return_tensors="pt", padding = True, return_attention_mask=True)
    label_list = []
    for label in data["binary_label"]: # 二分类
        label_list.append(label)
    result["labels"] = label_list
    return result

def process_data_5(data):
    speech_list = []
    for file_path in data["file_path"]:
        audio, _ = librosa.load(file_path, sr=16000)
        speech_list.append(audio)
    result = processor(speech_list, sampling_rate=16000, return_tensors="pt", padding = True, return_attention_mask=True)
    label_list = []
    for label in data["score_label"]:
        label_list.append(label)
    result["labels"] = label_list
    return result

# binary
# dataset = dataset.map(process_data_2, batched=True, batch_size=5, remove_columns=["file_path", "binary_label","gender","score_label"])
# dev_dataset = dev_dataset.map(process_data_2, batched=True, batch_size=5, remove_columns=["file_path", "binary_label","gender","score_label"])

# 5-class
dataset = dataset.map(process_data_5, batched=True, batch_size=5, remove_columns=["file_path", "binary_label","gender","score_label"])
dev_dataset = dev_dataset.map(process_data_5, batched=True, batch_size=5, remove_columns=["file_path", "binary_label","gender","score_label"])

model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=5)

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir = "./logs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=3e-5,
    save_steps=200,
    eval_steps=200,
    warmup_ratio=0.1,
    logging_steps=200,
    no_cuda=False
)

accuracy_metric = load_metric("accuracy")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")
f1_metric = load_metric("f1")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dev_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./model")

processor.save_pretrained("./model")

import json
import numpy as np
predictions = trainer.predict(dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
print(pred_labels)
print(dev_labels)
print(predictions.metrics)

pred_dict = {}
last_idx = 0
idx = 0
for participant_id, label_info in dev_labels.items():
    file_path = os.path.join('data/dev_uttr', f"spk_{participant_id}_uttr0.wav")
    last_idx = idx
    idx = 1
    while os.path.isfile(file_path):
        file_path = os.path.join('data/dev_uttr', f"spk_{participant_id}_uttr{idx}.wav")
        idx += 1

    idx -= 1
    pred_dict[participant_id] = pred_labels[last_idx:last_idx+idx].tolist()
print(pred_dict)

final_pred_dict = {}
for participant_id, pred_labels in pred_dict.items():
    final_pred_dict[int(participant_id)] = max(set(pred_labels), key=pred_labels.count)
print(final_pred_dict)

output_file = "audio_5label.json"
with open(output_file, "w") as f:
    json.dump(final_pred_dict, f)



