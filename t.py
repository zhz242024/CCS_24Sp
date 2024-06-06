import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from scipy.stats import ttest_ind

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_all_predictions():
    base_path = os.path.expanduser("~/CCS/code/")
    file_paths = {
        "au_2label": os.path.join(base_path, "au_2label.json"),
        "gaze_2label": os.path.join(base_path, "gaze_2label.json"),
        "audio_2label": os.path.join(base_path, "audio_2label.json"),
        "predictions_binary": os.path.join(base_path, "predictions_binary.json"),
    }

    predictions = {}
    for model, path in file_paths.items():
        predictions[model] = read_json(path)
    
    return predictions

# pred_matrix
def create_prediction_matrix(predictions, participant_ids, model_keys):
    prediction_matrix = []
    for model in model_keys:
        model_predictions = [predictions[model].get(str(pid), 0) for pid in participant_ids]
        prediction_matrix.append(model_predictions)
    return np.array(prediction_matrix)

# Binary weighted
def weighted_voting(prediction_matrix, accuracies, f1_scores, test_loss):
    accuracies = np.array(accuracies)
    f1_scores = np.array(f1_scores)
    test_loss = np.array(test_loss)

    performance_weights = (accuracies + f1_scores) / 2
    loss_weights = 1 / test_loss
    weights = performance_weights * loss_weights
    weights = normalize(weights.reshape(1, -1), norm='l1').flatten()

    weighted_predictions = np.zeros(prediction_matrix.shape[1])
    for i in range(len(weights)):
        weighted_predictions += weights[i] * prediction_matrix[i]
    
    final_predictions = (weighted_predictions >= 0.5).astype(int)
    return final_predictions

def get_true_labels(dev_split_file):
    labels_df = pd.read_csv(dev_split_file)
    true_labels_2label = labels_df['PHQ8_Binary'].values
    return true_labels_2label

# crossval
def cross_validate(predictions, true_labels, model_keys, accuracies, f1_scores, test_loss, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    accuracies_list, f1_scores_list, recalls_list = [], [], []

    for train_index, test_index in skf.split(np.zeros(len(true_labels)), true_labels):
        prediction_matrix = predictions[:, test_index]
        true_labels_fold = true_labels[test_index]

        final_predictions_weighted = weighted_voting(prediction_matrix, accuracies, f1_scores, test_loss)
        accuracy = accuracy_score(true_labels_fold, final_predictions_weighted)
        f1 = f1_score(true_labels_fold, final_predictions_weighted, average='macro')
        recall = recall_score(true_labels_fold, final_predictions_weighted, average='macro')

        accuracies_list.append(accuracy)
        f1_scores_list.append(f1)
        recalls_list.append(recall)

    return np.array(accuracies_list), np.array(f1_scores_list), np.array(recalls_list)

# t test
def significance_test(scores1, scores2):
    t_stat, p_value = ttest_ind(scores1, scores2)
    return t_stat, p_value

def main():
    predictions = get_all_predictions()
    participant_ids = list(predictions["predictions_binary"].keys())
    true_labels_2label = get_true_labels('~/CCS/dev_split_Depression_AVEC2017.csv')

    models_2label = ["au_2label", "gaze_2label", "audio_2label", "predictions_binary"]
    accuracies_2label = np.array([0.714286, 0.542857, 0.6, 0.5520164])
    f1_scores_2label = np.array([0.664021, 0.550049, 0.52963, 0.5317973255302682])
    test_loss_2label = np.array([0.6912, 0.7138, 0.7078, 1.4240])

    prediction_matrix_2label = create_prediction_matrix(predictions, participant_ids, models_2label)

    model_combinations = [
        ("au_2label", "gaze_2label"),
        ("au_2label", "audio_2label"),
        ("au_2label", "predictions_binary"),
        ("gaze_2label", "audio_2label"),
        ("audio_2label", "predictions_binary"),
        ("au_2label",),
        ("audio_2label",)
    ]

    cv_results = {}
    for combo in model_combinations:
        indices = [models_2label.index(model) for model in combo]
        selected_predictions = prediction_matrix_2label[indices]
        selected_accuracies = accuracies_2label[indices]
        selected_f1_scores = f1_scores_2label[indices]
        selected_test_loss = test_loss_2label[indices]
        
        accuracies, f1_scores, recalls = cross_validate(selected_predictions, true_labels_2label, combo, selected_accuracies, selected_f1_scores, selected_test_loss)
        cv_results[combo] = (accuracies, f1_scores, recalls)
        print(f"Combination: {combo}, Mean Accuracy: {np.mean(accuracies)}, Std: {np.std(accuracies)}, Mean F1: {np.mean(f1_scores)}, Std: {np.std(f1_scores)}")

    combos_to_compare = [
        (("au_2label", "gaze_2label"), ("au_2label", "audio_2label")),
        (("au_2label", "audio_2label"), ("au_2label", "predictions_binary")),
        (("au_2label", "predictions_binary"), ("gaze_2label", "audio_2label")),
        (("gaze_2label", "audio_2label"), ("audio_2label", "predictions_binary")),
        (("audio_2label", "predictions_binary"), ("au_2label",)),
        (("au_2label",), ("audio_2label",))
    ]

    print("\nT-tests between combinations:")
    for (combo1, combo2) in combos_to_compare:
        t_stat, p_value = significance_test(cv_results[combo1][1], cv_results[combo2][1])  # F1 score comparison
        print(f"T-test between {combo1} and {combo2}: T-statistic = {t_stat}, P-value = {p_value}")

if __name__ == "__main__":
    main()
