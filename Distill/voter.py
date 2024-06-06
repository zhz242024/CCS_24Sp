import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import normalize
from itertools import combinations

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_all_predictions():
    base_path = os.path.expanduser("~/CCS/code/")
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

# pred mat
def create_prediction_matrix(predictions, participant_ids, model_keys):
    prediction_matrix = []
    for model in model_keys:
        model_predictions = [predictions[model].get(str(pid), 0) for pid in participant_ids]
        prediction_matrix.append(model_predictions)
    return np.array(prediction_matrix)

# Bin weighted
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

# mult weighted
def weighted_voting_multi(prediction_matrix, accuracies, f1_scores, test_loss):
    return weighted_voting(prediction_matrix, accuracies, f1_scores, test_loss)

def get_true_labels(dev_split_file):
    labels_df = pd.read_csv(dev_split_file)
    true_labels_2label = labels_df['PHQ8_Binary'].values
    true_labels_5label = labels_df['PHQ8_Score'].apply(get_phq8_label).values
    return true_labels_2label, true_labels_5label

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

def evaluate_predictions(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    return accuracy, f1, recall

def main():
    accuracies_2label = np.array([0.714286, 0.542857, 0.6, 0.5520164])
    f1_scores_2label = np.array([0.664021, 0.550049, 0.52963, 0.5317973255302682])
    test_loss_2label = np.array([0.6912, 0.7138, 0.7078, 1.4240])

    accuracies_5label = np.array([0.314285, 0.314286, 0.142857, 0.2298998])
    f1_scores_5label = np.array([0.264208, 0.216326, 0.104635, 0.17623507129077787])
    test_loss_5label = np.array([1.6398, 1.583, 1.6171, 2.8289])

    predictions = get_all_predictions()
    
    participant_ids = list(predictions["predictions_binary"].keys())
    
    models_2label = ["au_2label", "gaze_2label", "audio_2label", "predictions_binary"]
    models_5label = ["au_5label", "gaze_5label", "audio_5label", "predictions_multi"]

    prediction_matrix_2label = create_prediction_matrix(predictions, participant_ids, models_2label)
    prediction_matrix_5label = create_prediction_matrix(predictions, participant_ids, models_5label)
    
    true_labels_2label, true_labels_5label = get_true_labels('~/CCS/dev_split_Depression_AVEC2017.csv')

    # 4-3

    print("\nAblation Studies for 2-Label Task (4 Choose 3)")
    for combo in ablation_combinations_2label:
        print(f"Evaluating models: {combo}")
        indices = [models_2label.index(model) for model in combo]
        selected_accuracies = accuracies_2label[indices]
        selected_f1_scores = f1_scores_2label[indices]
        selected_test_loss = test_loss_2label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_2label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

    print("\nAblation Studies for 5-Label Task (4 Choose 3)")
    for combo in ablation_combinations_5label:
        print(f"Evaluating models: {combo}")
        indices = [models_5label.index(model) for model in combo]
        selected_accuracies = accuracies_5label[indices]
        selected_f1_scores = f1_scores_5label[indices]
        selected_test_loss = test_loss_5label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting_multi(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_5label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

    # 3-2
    ablation_combinations_2label = list(combinations(models_2label, 2))
    ablation_combinations_5label = list(combinations(models_5label, 2))

    print("\nAblation Studies for 2-Label Task (3 Choose 2)")
    for combo in ablation_combinations_2label:
        print(f"Evaluating models: {combo}")
        indices = [models_2label.index(model) for model in combo]
        selected_accuracies = accuracies_2label[indices]
        selected_f1_scores = f1_scores_2label[indices]
        selected_test_loss = test_loss_2label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_2label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

    print("\nAblation Studies for 5-Label Task (3 Choose 2)")
    for combo in ablation_combinations_5label:
        print(f"Evaluating models: {combo}")
        indices = [models_5label.index(model) for model in combo]
        selected_accuracies = accuracies_5label[indices]
        selected_f1_scores = f1_scores_5label[indices]
        selected_test_loss = test_loss_5label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting_multi(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_5label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

    # 2-1
    ablation_combinations_2label = list(combinations(models_2label, 1))
    ablation_combinations_5label = list(combinations(models_5label, 1))

    print("\nAblation Studies for 2-Label Task (2 Choose 1)")
    for combo in ablation_combinations_2label:
        print(f"Evaluating models: {combo}")
        indices = [models_2label.index(model) for model in combo]
        selected_accuracies = accuracies_2label[indices]
        selected_f1_scores = f1_scores_2label[indices]
        selected_test_loss = test_loss_2label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_2label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

    print("\nAblation Studies for 5-Label Task (2 Choose 1)")
    for combo in ablation_combinations_5label:
        print(f"Evaluating models: {combo}")
        indices = [models_5label.index(model) for model in combo]
        selected_accuracies = accuracies_5label[indices]
        selected_f1_scores = f1_scores_5label[indices]
        selected_test_loss = test_loss_5label[indices]
        selected_predictions = create_prediction_matrix(predictions, participant_ids, combo)

        final_predictions_weighted = weighted_voting_multi(selected_predictions, selected_accuracies, selected_f1_scores, selected_test_loss)
        accuracy_weighted, f1_weighted, recall_weighted = evaluate_predictions(true_labels_5label, final_predictions_weighted)
        print("Weighted Voter - Accuracy: {:.4f}, F1: {:.4f}, Recall: {:.4f}".format(accuracy_weighted, f1_weighted, recall_weighted))

if __name__ == "__main__":
    main()
