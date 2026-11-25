"""
detector/tests_live_monitoring.py

Unit and Integration Tests for Live Monitoring Demonstration Feature.
Verifies login protection, live traffic adapter execution, Random Forest ML integration,
PredictionHistory logging, attack alert generation, user isolation, and graceful error handling.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory
from detector.live_monitor import execute_live_prediction, collect_local_traffic_features


class LiveMonitoringTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'monitortestuser'
        self.password = 'Pass12345!'
        self.user = User.objects.create_user(username=self.username, password=self.password)

        self.other_user = User.objects.create_user(username='otheruser', password=self.password)
        
        self.live_monitor_url = reverse('detector:live_monitor')
        self.live_sample_url = reverse('detector:live_monitor_sample')

    def test_live_monitor_page_login_required(self):
        """Unauthenticated user accessing live monitor page is redirected to login."""
        response = self.client.get(self.live_monitor_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)

    def test_live_monitor_page_authenticated(self):
        """Authenticated user can access the Live Monitoring page."""
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(self.live_monitor_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'detector/live_monitor.html')
        self.assertContains(response, 'Local Live Monitoring Demonstration')

    def test_live_monitor_sample_login_required(self):
        """Unauthenticated call to live monitor sample endpoint redirects to login."""
        response = self.client.get(self.live_sample_url)
        self.assertEqual(response.status_code, 302)

    def test_live_monitor_sample_normal_traffic(self):
        """Authenticated call to live monitor sample with normal mode returns success JSON."""
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(f"{self.live_sample_url}?mode=normal")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertTrue(data['available'])
        self.assertEqual(data['prediction'], 'NORMAL TRAFFIC')
        self.assertFalse(data['is_attack'])
        self.assertGreaterEqual(data['confidence'], 0.0)
        
        # Verify PredictionHistory was created for request.user
        history_rec = PredictionHistory.objects.filter(user=self.user).first()
        self.assertIsNotNone(history_rec)
        self.assertEqual(history_rec.prediction, 'NORMAL TRAFFIC')
        self.assertFalse(history_rec.is_threat)

    def test_live_monitor_sample_attack_simulation(self):
        """Authenticated call to live monitor sample with attack simulation mode detects attack and returns alert data."""
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(f"{self.live_sample_url}?mode=attack_simulation")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data['success'])
        self.assertTrue(data['is_attack'])
        self.assertEqual(data['prediction'], 'ATTACK DETECTED')
        self.assertIsNotNone(data['alert'])
        self.assertIn('ATTACK DETECTED', data['alert']['title'])
        
        # Verify PredictionHistory record created as a threat
        history_rec = PredictionHistory.objects.filter(user=self.user, is_threat=True).first()
        self.assertIsNotNone(history_rec)
        self.assertTrue(history_rec.is_threat)

    def test_live_monitor_user_isolation(self):
        """Verify live monitoring predictions are strictly isolated to the authenticated user."""
        self.client.login(username=self.username, password=self.password)
        self.client.get(f"{self.live_sample_url}?mode=normal")

        # Verify only monitortestuser has records, other_user has none
        self.assertEqual(PredictionHistory.objects.filter(user=self.user).count(), 1)
        self.assertEqual(PredictionHistory.objects.filter(user=self.other_user).count(), 0)

    def test_collect_local_traffic_features_labeling(self):
        """Verify data collector adapter properly labels source as Local Monitoring / Demonstration Data."""
        response = self.client.get(self.live_monitor_url)
        features, data_label = collect_local_traffic_features(response.wsgi_request, mode='normal')
        self.assertEqual(data_label, "Local Monitoring / Demonstration Data")
        self.assertIn('dur', features)
        self.assertIn('proto', features)
        self.assertEqual(len(features), 42)

    def test_network_monitor_route_removed(self):
        """Verify /network-monitor/ route is completely removed and returns 404."""
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/network-monitor/')
        self.assertEqual(response.status_code, 404)

    def test_live_monitor_auto_start_script_present(self):
        """Verify Live Monitor page contains auto-start script on DOMContentLoaded."""
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(self.live_monitor_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'DOMContentLoaded')
        self.assertContains(response, 'startMonitoring()')
