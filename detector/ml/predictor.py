"""
detector/ml/predictor.py

Intrusion Detection Predictor Service.
Loads `ml/artifacts/final_model.joblib` and `ml/artifacts/preprocessor.joblib`.
Performs input validation and runs single-sample or batch inference.
"""

import os
import pathlib
import datetime
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # Intrusion_Detection root
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

# The 42 required feature columns expected by the preprocessing pipeline
FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

# Numeric feature names
NUMERIC_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
    'ct_srv_dst', 'is_sm_ips_ports'
]

# Categorical feature names
CATEGORICAL_FEATURES = ['proto', 'service', 'state']

class IntrusionPredictor:
    _instance = None
    _model = None
    _preprocessor = None

    def __new__(cls):
        """Singleton pattern to cache model and preprocessor in memory."""
        if cls._instance is None:
            cls._instance = super(IntrusionPredictor, cls).__new__(cls)
            cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {PREPROCESSOR_PATH}")
            
        self._model = joblib.load(MODEL_PATH)
        self._preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"[IntrusionPredictor] Successfully cached {type(self._model).__name__} and preprocessor.")

    def validate_input(self, input_dict: dict):
        """
        Validate incoming user request dictionary.
        Returns (is_valid, cleaned_dict_or_error_list).
        """
        cleaned = {}
        errors = []

        # Ensure no forbidden fields are provided
        forbidden_fields = ['id', 'label', 'attack_cat']
        for field in forbidden_fields:
            if field in input_dict:
                errors.append(f"Forbidden field '{field}' must not be submitted.")

        # Validate numeric features
        for col in NUMERIC_FEATURES:
            val = input_dict.get(col, None)
            if val is None or str(val).strip() == "":
                errors.append(f"Missing required numeric field: '{col}'")
                continue
            try:
                num_val = float(val)
                if np.isnan(num_val) or np.isinf(num_val):
                    errors.append(f"Invalid numeric value for '{col}'")
                else:
                    cleaned[col] = num_val
            except (ValueError, TypeError):
                errors.append(f"Field '{col}' must be a valid number.")

        # Validate categorical features
        for col in CATEGORICAL_FEATURES:
            val = input_dict.get(col, None)
            if val is None or str(val).strip() == "":
                errors.append(f"Missing required categorical field: '{col}'")
            else:
                cleaned[col] = str(val).strip()

        if errors:
            return False, errors
        return True, cleaned

    def predict_sample(self, input_dict: dict) -> dict:
        """
        Run intrusion prediction on a sample dictionary.
        """
        is_valid, val_result = self.validate_input(input_dict)
        if not is_valid:
            return {
                'success': False,
                'errors': val_result,
                'prediction': 'ERROR'
            }

        # Convert dictionary to single-row pandas DataFrame
        df_input = pd.DataFrame([val_result])
        
        # Ensure column order matches FEATURE_COLUMNS
        df_input = df_input[FEATURE_COLUMNS]

        # Transform features using preprocessor (FIT IS NEVER CALLED HERE)
        X_processed = self._preprocessor.transform(df_input)

        # Predict
        pred_class = int(self._model.predict(X_processed)[0])
        
        # Calculate probabilities
        confidence = 0.0
        prob_attack = 0.0
        prob_normal = 0.0

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X_processed)[0]
            prob_normal = float(probs[0])
            prob_attack = float(probs[1]) if len(probs) > 1 else 0.0
            confidence = float(probs[pred_class]) * 100.0
        else:
            confidence = 100.0

        is_attack = bool(pred_class == 1)
        pred_label = "ATTACK DETECTED" if is_attack else "NORMAL TRAFFIC"

        explanation = (
            f"The Random Forest model classified the submitted network traffic flow as {pred_label} "
            f"with a confidence of {confidence:.2f}% based on pattern analysis of packet rates, "
            f"TTL headers, and payload attributes."
        )

        return {
            'success': True,
            'prediction': pred_label,
            'is_attack': is_attack,
            'confidence': round(confidence, 2),
            'prob_normal': round(prob_normal * 100, 2),
            'prob_attack': round(prob_attack * 100, 2),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'explanation': explanation,
            'errors': []
        }
