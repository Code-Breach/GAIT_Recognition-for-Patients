#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import os
import pandas as pd
def load_smpl_data(person_id, npz_root, walk_df):
    features = []

    # Filter the walks for this person
    person_walks = walk_df[walk_df['ID'] == person_id]

    for _, row in person_walks.iterrows():
        file_path = os.path.join(npz_root, row['file_path'])
        if not os.path.exists(file_path):
            continue
        try:
            data = np.load(file_path)
            frame_feats = []

            if 'pose' in data:
                frame_feats.append(data['pose'].mean(axis=0) if data['pose'].ndim > 1 else data['pose'])  # (72,)
            if 'shape' in data:
                frame_feats.append(data['shape'].mean(axis=0) if data['shape'].ndim > 1 else data['shape'])  # (10,)
            if 'global_t' in data:
                frame_feats.append(data['global_t'].mean(axis=0))  # (3,)
            if 'focal_l' in data:
                frame_feats.append([np.mean(data['focal_l'])])  # scalar
            if 'pred_joints' in data:
                frame_feats.append(data['pred_joints'].mean(axis=(0, 1)))  # (3,)

            # Optionally add metadata like viewpoint, variation
            frame_feats.append([row['viewpoint']])
            frame_feats.append([hash(row['variation']) % 1000])  # or use OneHotEncoding if needed

            features.append(np.concatenate(frame_feats))

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue

    if len(features) == 0:
        raise ValueError(f"No valid files found for ID {person_id}")

    return np.mean(features, axis=0)  # aggregate over all variations


# In[4]:


# Load walk metadata from CSV
def load_walk_metadata(walk_csv):
    walk_df = pd.read_csv(walk_csv)
    walk_df['file_path'] = walk_df.apply(lambda row: f"{row['ID']}/{row['file_id'].replace(':', '_')}.npz", axis=1)
    return walk_df

# Load GHQ labels from CSV
def load_ghq_labels(csv_file):
    ghq_data = pd.read_csv(csv_file)
    return ghq_data.set_index('ID')['GHQ_Label'].to_dict()  # Create a mapping from person_id to ghq_label


# In[5]:


def process_data(npz_root, ghq_csv, walk_csv):
    walk_df = load_walk_metadata(walk_csv)
    ghq_labels = load_ghq_labels(ghq_csv)

    all_features, all_labels = [], []

    for person_id in ghq_labels.keys():
        try:
            feats = load_smpl_data(person_id, npz_root, walk_df)
            all_features.append(feats)
            all_labels.append(ghq_labels[person_id])
        except Exception as e:
            print(f"Skipping ID {person_id}: {e}")

    return np.array(all_features), np.array(all_labels)


# In[6]:


# Example usage
walk_df = 'walks.csv'
npz_folder = 'smpl'  # Replace with the path to your npz folder
ghq_csv = 'subset.csv'  # Replace with the path to your GHQ labels CSV


# In[7]:





# In[8]:
from sklearn.preprocessing import LabelEncoder





# In[15]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def try_xgboost(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=280)

    model = XGBClassifier( eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return model, scaler




# In[10]:


import numpy as np
import os

def extract_smpl_features(npz_path, viewpoint=0, variation_hash=0):
    """
    Load a single .npz and turn it into the same feature‐vector you used for training.
    If you don’t have viewpoint/variation info, you can default them (e.g. 0).
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found")

    data = np.load(npz_path)
    feats = []

    # pose: (72,)
    if 'pose' in data:
        p = data['pose']
        feats.append(p.mean(axis=0) if p.ndim > 1 else p)

    # shape: (10,)
    if 'shape' in data:
        s = data['shape']
        feats.append(s.mean(axis=0) if s.ndim > 1 else s)

    # global translation: (3,)
    if 'global_t' in data:
        feats.append(data['global_t'].mean(axis=0))

    # focal length (scalar)
    if 'focal_l' in data:
        feats.append([np.mean(data['focal_l'])])

    # predicted joints: flatten mean over frames & joints
    if 'pred_joints' in data:
        # data['pred_joints'] shape e.g. (num_frames, num_joints, 3)
        feats.append(data['pred_joints'].mean(axis=(0,1)))

    # add—or default—your “metadata” slots:
    feats.append([viewpoint])
    feats.append([variation_hash])

    # final 1D vector
    return np.concatenate(feats)


# In[16]:


from joblib import dump







if __name__ == "__main__":
    features, labels = process_data(npz_root="smpl", ghq_csv="subset.csv", walk_csv="walks.csv")
    model, scaler = try_xgboost(features, labels, show_report=True)

    # Assuming labels are like ['Major Distress', 'Typical', ...]
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    dump(model, 'model.joblib')
    dump(le,       'label_encoder.joblib')
    dump(scaler, 'scaler.joblib')# In[ ]:





# 2. Support Vector Machine (SVM)

# In[43]:


