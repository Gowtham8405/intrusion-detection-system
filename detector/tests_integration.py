"""
detector/tests_integration.py

Functional integration tests for Step 7: Django ML Integration & Prediction.
Executes against actual validation dataset samples (NOT test dataset).
"""

import os
import pathlib
import pandas as pd
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory
from detector.ml.predictor import IntrusionPredictor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL_PATH = PROJECT_ROOT / "datasets" / "validation" / "validation.csv"

class IntrusionDetectionIntegrationTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='integration_user', password='Password123!')
        self.client.login(username='integration_user', password='Password123!')
        
        self.predictor = IntrusionPredictor()
        self.val_df = pd.read_csv(VAL_PATH)
        
        # Prepare sample data (excluding forbidden columns)
        normal_raw = self.val_df[self.val_df['label'] == 0].iloc[0].to_dict()
        attack_raw = self.val_df[self.val_df['label'] == 1].iloc[0].to_dict()
        
        self.normal_input = {k: str(v) for k, v in normal_raw.items() if k in FEATURE_COLUMNS}
        self.attack_input = {k: str(v) for k, v in attack_raw.items() if k in FEATURE_COLUMNS}

    def test_01_predictor_caching_and_loading(self):
        """Verify model and preprocessor artifacts load and cache correctly."""
        self.assertIsNotNone(self.predictor._model)
        self.assertIsNotNone(self.predictor._preprocessor)
        self.assertEqual(type(self.predictor._model).__name__, "RandomForestClassifier")

    def test_02_normal_sample_prediction(self):
        """Test prediction service on actual validation Normal sample."""
        res = self.predictor.predict_sample(self.normal_input)
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], "NORMAL TRAFFIC")
        self.assertFalse(res['is_attack'])
        self.assertGreaterEqual(res['confidence'], 50.0)

    def test_03_attack_sample_prediction(self):
        """Test prediction service on actual validation Attack sample."""
        res = self.predictor.predict_sample(self.attack_input)
        self.assertTrue(res['success'])
        self.assertEqual(res['prediction'], "ATTACK DETECTED")
        self.assertTrue(res['is_attack'])
        self.assertGreaterEqual(res['confidence'], 50.0)

    def test_04_missing_field_rejection(self):
        """Test that missing required fields are rejected safely."""
        incomplete = self.normal_input.copy()
        del incomplete['dur']
        res = self.predictor.predict_sample(incomplete)
        self.assertFalse(res['success'])
        self.assertIn("Missing required numeric field: 'dur'", res['errors'])

    def test_05_invalid_numeric_rejection(self):
        """Test that invalid non-numeric inputs are rejected safely."""
        invalid = self.normal_input.copy()
        invalid['spkts'] = 'invalid_number_abc'
        res = self.predictor.predict_sample(invalid)
        self.assertFalse(res['success'])
        self.assertIn("Field 'spkts' must be a valid number.", res['errors'])

    def test_06_forbidden_field_rejection(self):
        """Test that forbidden fields (id, label, attack_cat) are rejected."""
        forbidden = self.normal_input.copy()
        forbidden['label'] = '1'
        res = self.predictor.predict_sample(forbidden)
        self.assertFalse(res['success'])
        self.assertIn("Forbidden field 'label' must not be submitted.", res['errors'])

    def test_07_django_post_view_normal(self):
        """Test Django POST request view with Normal sample."""
        response = self.client.post(reverse('detector:detect'), self.normal_input)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "NORMAL TRAFFIC")
        self.assertEqual(PredictionHistory.objects.count(), 1)
        rec = PredictionHistory.objects.first()
        self.assertFalse(rec.is_threat)
        self.assertEqual(rec.prediction, "NORMAL TRAFFIC")
        self.assertEqual(rec.user, self.user)

    def test_08_django_post_view_attack(self):
        """Test Django POST request view with Attack sample."""
        response = self.client.post(reverse('detector:detect'), self.attack_input)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ATTACK DETECTED")
        rec = PredictionHistory.objects.first()
        self.assertTrue(rec.is_threat)
        self.assertEqual(rec.prediction, "ATTACK DETECTED")
        self.assertEqual(rec.user, self.user)
