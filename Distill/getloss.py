import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_all_predictions():
    base_path = os.path.expanduser("~/CCS/code/label_result/au_gaze_audio_result/")
    file_paths = {
        "au_2label": os.path.join(base_path, "au_2label.json"),
        "au_5label": os.path.join(base_path, "au_5label.json"),
        "gaze_2label": os.path.join(base_path, "gaze_2label.json"),
        "gaze_5label": os.path.join(base_path, "gaze_5label.json"),
        "audio_2label": os.path.join(base_path, "audio_2label.json"),
        "audio_5label": os.path.join(base_path, "audio_5label.json"),
        "predictions_binary": os.path.join(base_path, "predictions_binary.json"),
        "predictions_multi": os.path.join(base_path, "predictions_multi.json"),
    }

    predictions = {}
    for model, path in file_paths.items():
        predictions[model] = read_json(path)
    
    return predictions

def get_true_labels(dev_split_file):
    labels_df = pd.read_csv(dev_split_file)
    true_binary_labels = labels_df['PHQ8_Binary'].values
    true_multi_labels = labels_df['PHQ8_Score'].apply(get_phq8_label).values
    return true_binary_labels, true_multi_labels

def get_phq8_label(score):
    if score <= 4:
        return 0
    elif score <= 9:
        return 1
    elif score <= 14:
        return 2
    elif score <= 19:
        return 3
    else:
        return 4

def compute_binary_loss(true_labels, pred_labels):
    binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return binary_loss(tf.convert_to_tensor(true_labels, dtype=tf.float32), tf.convert_to_tensor(pred_labels, dtype=tf.float32)).numpy()

def compute_multi_loss(true_labels, pred_labels):
    true_labels = tf.convert_to_tensor(true_labels, dtype=tf.int32)
    pred_labels = tf.convert_to_tensor(pred_labels, dtype=tf.float32)
    multi_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    return multi_loss(true_labels, pred_labels).numpy()

def main():
    predictions = get_all_predictions()
    
    participant_ids = list(predictions["predictions_binary"].keys())
    
    true_binary_labels, true_multi_labels = get_true_labels('~/CCS/dev_split_Depression_AVEC2017.csv')

    binary_pred_labels = {
        "au_2label": np.array([predictions["au_2label"].get(str(pid), 0) for pid in participant_ids]),
        "gaze_2label": np.array([predictions["gaze_2label"].get(str(pid), 0) for pid in participant_ids]),
        "audio_2label": np.array([predictions["audio_2label"].get(str(pid), 0) for pid in participant_ids]),
        "predictions_binary": np.array([predictions["predictions_binary"].get(str(pid), 0) for pid in participant_ids])
    }

    multi_pred_labels = {
        "au_5label": np.array([predictions["au_5label"].get(str(pid), 0) for pid in participant_ids]),
        "gaze_5label": np.array([predictions["gaze_5label"].get(str(pid), 0) for pid in participant_ids]),
        "audio_5label": np.array([predictions["audio_5label"].get(str(pid), 0) for pid in participant_ids]),
        "predictions_multi": np.array([predictions["predictions_multi"].get(str(pid), 0) for pid in participant_ids])
    }

    num_classes = 5
    for key in multi_pred_labels:
        multi_pred_labels[key] = tf.keras.utils.to_categorical(multi_pred_labels[key], num_classes=num_classes)

    print("True Binary Labels (sample):", true_binary_labels[:5])
    print("Binary Predictions (sample):", binary_pred_labels["au_2label"][:5])
    
    print("True Multi Labels (sample):", true_multi_labels[:5])
    print("Multi Predictions (sample):", multi_pred_labels["au_5label"][:5])

    for key in binary_pred_labels:
        print(f"{key} Binary Predictions (sample):", binary_pred_labels[key][:5])
    for key in multi_pred_labels:
        print(f"{key} Multi Predictions (sample):", multi_pred_labels[key][:5])

    binary_losses = {}
    for key, pred_labels in binary_pred_labels.items():
        pred_labels = np.expand_dims(pred_labels, axis=-1)
        pred_labels = np.clip(pred_labels, 0, 1)  
        binary_losses[key] = compute_binary_loss(true_binary_labels, pred_labels)
    multi_losses = {}
    for key, pred_labels in multi_pred_labels.items():
        multi_losses[key] = compute_multi_loss(true_multi_labels, pred_labels)
    for key, loss in binary_losses.items():
        print(f"Test Loss for {key}: {loss}")
    for key, loss in multi_losses.items():
        print(f"Test Loss for {key}: {loss}")

if __name__ == "__main__":
    main()






