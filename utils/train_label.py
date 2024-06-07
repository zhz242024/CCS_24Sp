import pickle
import pandas as pd

def label_extraction():
    label_csv = pd.read_csv('data/test_split_Depression_AVEC2017.csv')
    label_dict = {}
    for i in range(len(label_csv)):
        # label_dict[label_csv['Participant_ID'][i]] = [label_csv['PHQ8_Binary'][i], label_csv['PHQ8_Score'][i], label_csv['Gender'][i]]
        label_dict[label_csv['participant_ID'][i]] = label_csv['Gender'][i]
    with open('feature/test_label.pickle', 'wb') as handle:
        pickle.dump(label_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)