import pandas as pd
import os
import shutil

dataset_dir = os.path.expanduser('~/CCS/DATASET')
train_dir = os.path.join(dataset_dir, 'train')
dev_dir = os.path.join(dataset_dir, 'dev')
test_dir = os.path.join(dataset_dir, 'test')

# Create train, dev, and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
train_split_path = os.path.join('/home/ucloud/CCS/train_split_Depression_AVEC2017.csv')
dev_split_path = os.path.join('/home/ucloud/CCS/dev_split_Depression_AVEC2017.csv')
test_split_path = os.path.join('/home/ucloud/CCS/test_split_Depression_AVEC2017.csv')
train_split = pd.read_csv(train_split_path)
dev_split = pd.read_csv(dev_split_path)
test_split = pd.read_csv(test_split_path)

# Print column names for debugging
print("Train Split Columns:", train_split.columns)
print("Dev Split Columns:", dev_split.columns)
print("Test Split Columns:", test_split.columns)

# Ensure the 'Participant_ID' and 'participant_ID' columns exist and handle whitespace
train_split.columns = train_split.columns.str.strip()
dev_split.columns = dev_split.columns.str.strip()
test_split.columns = test_split.columns.str.strip()

# Extract session IDs based on the correct column names
if 'Participant_ID' in train_split.columns:
    train_sessions = train_split['Participant_ID'].tolist()
else:
    raise KeyError("Column 'Participant_ID' not found in train split")

if 'Participant_ID' in dev_split.columns:
    dev_sessions = dev_split['Participant_ID'].tolist()
else:
    raise KeyError("Column 'Participant_ID' not found in dev split")

if 'participant_ID' in test_split.columns:
    test_sessions = test_split['participant_ID'].tolist()
else:
    raise KeyError("Column 'participant_ID' not found in test split")

def copy_files(session_ids, dest_dir):
    for session_id in session_ids:
        session_dir = os.path.join(dataset_dir, f'{session_id}_P')
        if os.path.exists(session_dir):
            shutil.copytree(session_dir, os.path.join(dest_dir, f'{session_id}_P'), dirs_exist_ok=True)

# Copy files
copy_files(train_sessions, train_dir)
copy_files(dev_sessions, dev_dir)
copy_files(test_sessions, test_dir)

print("Data successfully split")



