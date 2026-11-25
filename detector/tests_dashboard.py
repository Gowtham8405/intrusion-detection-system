"""
detector/tests_dashboard.py

Automated Dashboard Tests for Step 10.
Tests access control, user-specific data isolation, statistical calculations, recent predictions limit, and empty state rendering.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory


class UserDashboardTests(TestCase):

    def setUp(self):
        self.client_a = Client()
        self.client_b = Client()

        self.user_a = User.objects.create_user(username='dash_user_a', password='Password123!')
        self.user_b = User.objects.create_user(username='dash_user_b', password='Password123!')

        self.client_a.login(username='dash_user_a', password='Password123!')
        self.client_b.login(username='dash_user_b', password='Password123!')

    def test_01_dashboard_requires_authentication(self):
        """TEST 1: Unauthenticated request to /dashboard/ redirects to login."""
        unauth_client = Client()
        response = unauth_client.get(reverse('detector:dashboard'))
        self.assertRedirects(response, f"{reverse('detector:login')}?next={reverse('detector:dashboard')}")

    def test_02_authenticated_user_can_access_dashboard(self):
        """TEST 2: Authenticated user can access /dashboard/ with status 200."""
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Welcome back, dash_user_a!")

    def test_03_dashboard_shows_correct_total_prediction_count(self):
        """TEST 3: Dashboard context displays exact total prediction count."""
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.context['total_predictions'], 2)

    def test_04_dashboard_shows_correct_normal_count(self):
        """TEST 4: Dashboard context displays exact normal traffic count."""
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.context['normal_count'], 2)

    def test_05_dashboard_shows_correct_attack_count(self):
        """TEST 5: Dashboard context displays exact attack count."""
        PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", is_threat=True)
        PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.context['attacks_count'], 2)

    def test_06_dashboard_calculates_threat_percentage_correctly(self):
        """TEST 6: Dashboard calculates threat percentage accurately (e.g. 1 attack out of 4 = 25.0%)."""
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.context['threat_percentage'], 25.0)

    def test_07_dashboard_handles_zero_predictions_safely(self):
        """TEST 7: Dashboard handles 0 predictions safely without division-by-zero error."""
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_predictions'], 0)
        self.assertEqual(response.context['threat_percentage'], 0.0)
        self.assertContains(response, "Welcome to Your Intrusion Detection Dashboard")

    def test_08_dashboard_shows_only_current_user_records(self):
        """TEST 8: Dashboard shows only the current user's records in recent list."""
        rec_a = PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        rec_b = PredictionHistory.objects.create(user=self.user_b, prediction="ATTACK DETECTED", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        recent_records = list(response.context['recent_predictions'])
        self.assertIn(rec_a, recent_records)
        self.assertNotIn(rec_b, recent_records)

    def test_09_user_a_dashboard_does_not_include_user_b_statistics(self):
        """TEST 9: User A dashboard statistics do not count User B's predictions."""
        for _ in range(3):
            PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        for _ in range(7):
            PredictionHistory.objects.create(user=self.user_b, prediction="ATTACK DETECTED", is_threat=True)
        
        response_a = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(response_a.context['total_predictions'], 3)
        self.assertEqual(response_a.context['attacks_count'], 0)

        response_b = self.client_b.get(reverse('detector:dashboard'))
        self.assertEqual(response_b.context['total_predictions'], 7)
        self.assertEqual(response_b.context['attacks_count'], 7)

    def test_10_recent_predictions_show_maximum_5_records(self):
        """TEST 10: Recent predictions list contains at most 5 items."""
        for i in range(8):
            PredictionHistory.objects.create(user=self.user_a, prediction=f"PRED_{i}", is_threat=False)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertEqual(len(response.context['recent_predictions']), 5)

    def test_11_recent_predictions_ordered_newest_first(self):
        """TEST 11: Recent predictions are ordered descending by timestamp."""
        r1 = PredictionHistory.objects.create(user=self.user_a, prediction="FIRST", is_threat=False)
        r2 = PredictionHistory.objects.create(user=self.user_a, prediction="SECOND", is_threat=False)
        response = self.client_a.get(reverse('detector:dashboard'))
        recent = list(response.context['recent_predictions'])
        self.assertEqual(recent[0], r2)
        self.assertEqual(recent[1], r1)

    def test_12_dashboard_does_not_expose_other_user_prediction_info(self):
        """TEST 12: User A's dashboard HTML does not contain User B's secret prediction string."""
        PredictionHistory.objects.create(user=self.user_b, prediction="ATTACK_USER_B_SECRET_STRING", is_threat=True)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertNotContains(response, "ATTACK_USER_B_SECRET_STRING")

    def test_13_existing_history_link_accessible(self):
        """TEST 13: Dashboard contains a link to full history page."""
        PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", is_threat=False)
        response = self.client_a.get(reverse('detector:dashboard'))
        self.assertContains(response, reverse('detector:history'))
