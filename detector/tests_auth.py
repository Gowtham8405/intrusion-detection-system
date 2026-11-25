"""
detector/tests_auth.py

Automated Authentication & User Data Isolation Tests for Step 8.
Tests Django User Registration, Login, Logout, Session Protection, and History Data Isolation.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory
from detector.ml.predictor import IntrusionPredictor, FEATURE_COLUMNS
import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL_PATH = PROJECT_ROOT / "datasets" / "validation" / "validation.csv"


class UserAuthenticationAndIsolationTests(TestCase):

    def setUp(self):
        self.client = Client()
        # Create test users
        self.user_a = User.objects.create_user(username='test_user_a', password='Password123!', email='usera@example.com')
        self.user_b = User.objects.create_user(username='test_user_b', password='Password123!', email='userb@example.com')

        # Sample input
        val_df = pd.read_csv(VAL_PATH)
        normal_raw = val_df[val_df['label'] == 0].iloc[0].to_dict()
        attack_raw = val_df[val_df['label'] == 1].iloc[0].to_dict()

        self.normal_input = {k: str(v) for k, v in normal_raw.items() if k in FEATURE_COLUMNS}
        self.attack_input = {k: str(v) for k, v in attack_raw.items() if k in FEATURE_COLUMNS}

    def test_01_registration_page_loads(self):
        """TEST 1: Registration page loads with status 200."""
        response = self.client.get(reverse('detector:register'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create User Account")

    def test_02_valid_registration_creates_user(self):
        """TEST 2: Valid registration creates a new user in database with hashed password."""
        response = self.client.post(reverse('detector:register'), {
            'username': 'new_user_123',
            'email': 'newuser@example.com',
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!'
        })
        self.assertRedirects(response, reverse('detector:login'))
        self.assertTrue(User.objects.filter(username='new_user_123').exists())
        user = User.objects.get(username='new_user_123')
        self.assertTrue(user.check_password('SecurePassword123!'))

    def test_03_duplicate_username_rejected(self):
        """TEST 3: Duplicate username registration is rejected with error."""
        response = self.client.post(reverse('detector:register'), {
            'username': 'test_user_a',
            'email': 'duplicate@example.com',
            'password': 'SecurePassword123!',
            'confirm_password': 'SecurePassword123!'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Username already exists.")

    def test_04_password_mismatch_rejected(self):
        """TEST 4: Password mismatch during registration is rejected."""
        response = self.client.post(reverse('detector:register'), {
            'username': 'mismatch_user',
            'email': 'mismatch@example.com',
            'password': 'Password123!',
            'confirm_password': 'DifferentPassword456!'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Passwords do not match.")

    def test_05_valid_login_succeeds(self):
        """TEST 5: Valid login succeeds and establishes a session."""
        response = self.client.post(reverse('detector:login'), {
            'username': 'test_user_a',
            'password': 'Password123!'
        })
        self.assertRedirects(response, reverse('detector:detect'))
        self.assertIn('_auth_user_id', self.client.session)

    def test_06_invalid_login_fails(self):
        """TEST 6: Invalid login fails and shows error message."""
        response = self.client.post(reverse('detector:login'), {
            'username': 'test_user_a',
            'password': 'WrongPassword!'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid username or password.")
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_07_logout_works(self):
        """TEST 7: Logout ends the session and redirects to login."""
        self.client.login(username='test_user_a', password='Password123!')
        response = self.client.get(reverse('detector:logout'))
        self.assertRedirects(response, reverse('detector:login'))
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_08_unauthenticated_user_blocked_from_detect(self):
        """TEST 8: Unauthenticated user cannot access protected detection page."""
        response = self.client.get(reverse('detector:detect'))
        self.assertRedirects(response, f"{reverse('detector:login')}?next={reverse('detector:detect')}")

    def test_09_unauthenticated_user_blocked_from_history(self):
        """TEST 9: Unauthenticated user cannot access protected history page."""
        response = self.client.get(reverse('detector:history'))
        self.assertRedirects(response, f"{reverse('detector:login')}?next={reverse('detector:history')}")

    def test_10_authenticated_user_can_access_history(self):
        """TEST 10: Authenticated user can access history page."""
        self.client.login(username='test_user_a', password='Password123!')
        response = self.client.get(reverse('detector:history'))
        self.assertEqual(response.status_code, 200)

    def test_11_prediction_saved_with_authenticated_user(self):
        """TEST 11: Prediction record is saved with request.user ForeignKey."""
        self.client.login(username='test_user_a', password='Password123!')
        response = self.client.post(reverse('detector:detect'), self.normal_input)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        record = PredictionHistory.objects.first()
        self.assertEqual(record.user, self.user_a)

    def test_12_user_a_cannot_see_user_b_history(self):
        """TEST 12: User A cannot see User B's prediction history."""
        # Create record for User B
        PredictionHistory.objects.create(
            user=self.user_b,
            prediction="ATTACK DETECTED",
            confidence=99.5,
            is_threat=True
        )

        # Login as User A
        self.client.login(username='test_user_a', password='Password123!')
        response = self.client.get(reverse('detector:history'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "#1")  # User B's record should not appear in User A's history view

    def test_13_user_b_cannot_see_user_a_history(self):
        """TEST 13: User B cannot see User A's prediction history."""
        # Create record for User A
        PredictionHistory.objects.create(
            user=self.user_a,
            prediction="NORMAL TRAFFIC",
            confidence=98.1,
            is_threat=False
        )

        # Login as User B
        self.client.login(username='test_user_b', password='Password123!')
        response = self.client.get(reverse('detector:history'))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "NORMAL TRAFFIC")  # User A's record hidden from User B

    def test_14_user_isolation_query(self):
        """TEST 14: PredictionHistory.objects.filter(user=request.user) enforces exact data isolation."""
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", confidence=95.0)
        PredictionHistory.objects.create(user=self.user_b, prediction="ATTACK DETECTED", confidence=96.0)

        user_a_records = PredictionHistory.objects.filter(user=self.user_a)
        user_b_records = PredictionHistory.objects.filter(user=self.user_b)

        self.assertEqual(user_a_records.count(), 1)
        self.assertEqual(user_a_records.first().prediction, "NORMAL TRAFFIC")
        self.assertEqual(user_b_records.count(), 1)
        self.assertEqual(user_b_records.first().prediction, "ATTACK DETECTED")

    def test_15_csrf_token_present(self):
        """TEST 15: CSRF protection token is present in auth forms."""
        response_login = self.client.get(reverse('detector:login'))
        response_register = self.client.get(reverse('detector:register'))
        self.assertContains(response_login, "csrfmiddlewaretoken")
        self.assertContains(response_register, "csrfmiddlewaretoken")
