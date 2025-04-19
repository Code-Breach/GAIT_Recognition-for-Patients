import sys
import numpy as np
from joblib import load
from code1 import extract_smpl_features  # wherever you put it

def predict_from_npz(npz_path, viewpoint=0, variation_hash=0):
    # 1. extract
    feats = extract_smpl_features(npz_path, viewpoint, variation_hash).reshape(1, -1)

    # 2. load artifacts
    scaler        = load('scaler.joblib')
    label_encoder = load('label_encoder.joblib')
    model         = load('xgb_model.joblib')

    # 3. scale & predict
    feats_scaled = scaler.transform(feats)
    y_pred_enc   = model.predict(feats_scaled)

    # 4. decode back to original GHQ_Label
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    return y_pred[0]

if __name__ == "__main__":
    # Usage: python predict.py path/to/file.npz [viewpoint] [variation_hash]
    if len(sys.argv) < 2:
        print("Usage: python predict.py <npz_path> [viewpoint:int] [variation_hash:int]")
        sys.exit(1)

    npz_file = sys.argv[1]
    vp        = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    vh        = int(sys.argv[3]) if len(sys.argv) >= 4 else 0

    label = predict_from_npz(npz_file, vp, vh)
    print(f"Predicted GHQ_Label: {label}")
