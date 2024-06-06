import numpy as np
import pandas as pd

feature_file = np.load('/home/ucloud/CCS/code/DistillBert/train_features.npy.npz')
features = feature_file['features']
patient_ids = feature_file['patient_ids']
print(f"Features shape: {features.shape}")
print(f"Patient IDs shape: {patient_ids.shape}")
df = pd.DataFrame(patient_ids, columns=['Patient_ID'])
unique_patient_ids = df['Patient_ID'].unique()
print(f"Unique Patient IDs: {len(unique_patient_ids)}")
patient_id_counts = df['Patient_ID'].value_counts()
print(patient_id_counts)
for patient_id in unique_patient_ids:
    patient_features = features[df['Patient_ID'] == patient_id]
    print(f"Patient ID: {patient_id}, Number of Features: {len(patient_features)}")
    if len(patient_features) > 0:
        print(f"Features for {patient_id}:\n{patient_features[:5]}\n")

