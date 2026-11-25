from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory


class UIAndErrorHandlingTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user_a = User.objects.create_user(username='usera', password='password123')
        self.user_b = User.objects.create_user(username='userb', password='password123')
        self.dashboard_url = reverse('detector:dashboard')
        self.analytics_url = reverse('detector:analytics')
        self.history_url = reverse('detector:history')
        self.detect_url = reverse('detector:detect')
        self.login_url = reverse('detector:login')
        self.register_url = reverse('detector:register')

    # Test 1: 404 page behavior
    def test_404_page_behavior(self):
        response = self.client.get('/invalid-page-does-not-exist/')
        self.assertEqual(response.status_code, 404)

    # Test 2: Protected dashboard redirects unauthenticated users
    def test_protected_dashboard_redirects_unauthenticated(self):
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(self.login_url, response.url)

    # Test 3: Protected analytics redirects unauthenticated users
    def test_protected_analytics_redirects_unauthenticated(self):
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(self.login_url, response.url)

    # Test 4: Protected history redirects unauthenticated users
    def test_protected_history_redirects_unauthenticated(self):
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(self.login_url, response.url)

    # Test 5: Invalid login handling
    def test_invalid_login_handling(self):
        response = self.client.post(self.login_url, {
            'username': 'usera',
            'password': 'wrongpassword'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid username or password.")

    # Test 6: Registration validation handling
    def test_registration_validation_mismatched_password(self):
        response = self.client.post(self.register_url, {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123',
            'confirm_password': 'mismatchpassword'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Passwords do not match.")

    # Test 7: Detection validation handling
    def test_detection_validation_handling(self):
        self.client.login(username='usera', password='password123')
        # Submit invalid numerical value
        post_data = {'dur': 'invalid_string', 'proto': 'tcp', 'service': 'http', 'state': 'FIN'}
        response = self.client.post(self.detect_url, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Validation Error")

    # Test 8: Empty dashboard state
    def test_empty_dashboard_state(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Welcome to Your Intrusion Detection Dashboard")

    # Test 9: Empty analytics state
    def test_empty_analytics_state(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Analytics Data Yet")

    # Test 10: Empty history state
    def test_empty_history_state(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Prediction History Yet")

    # Test 11: Own prediction delete remains functional
    def test_own_prediction_delete_remains_functional(self):
        record = PredictionHistory.objects.create(
            user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False
        )
        self.client.login(username='usera', password='password123')
        delete_url = reverse('detector:delete_prediction', kwargs={'record_id': record.id})
        response = self.client.post(delete_url, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(PredictionHistory.objects.filter(id=record.id).exists())

    # Test 12: Cross-user prediction delete remains blocked
    def test_cross_user_prediction_delete_blocked(self):
        record_b = PredictionHistory.objects.create(
            user=self.user_b, prediction='Attack', confidence=98.0, is_threat=True
        )
        self.client.login(username='usera', password='password123')
        delete_url = reverse('detector:delete_prediction', kwargs={'record_id': record_b.id})
        response = self.client.post(delete_url)
        self.assertEqual(response.status_code, 404)
        self.assertTrue(PredictionHistory.objects.filter(id=record_b.id).exists())

    # Test 13: GET delete remains blocked
    def test_get_delete_remains_blocked(self):
        record = PredictionHistory.objects.create(
            user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False
        )
        self.client.login(username='usera', password='password123')
        delete_url = reverse('detector:delete_prediction', kwargs={'record_id': record.id})
        response = self.client.get(delete_url)
        self.assertEqual(response.status_code, 405)
        self.assertTrue(PredictionHistory.objects.filter(id=record.id).exists())

    # Test 14: Clear history remains POST-only
    def test_clear_history_remains_post_only(self):
        PredictionHistory.objects.create(
            user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False
        )
        self.client.login(username='usera', password='password123')
        clear_url = reverse('detector:clear_history')
        response = self.client.get(clear_url)
        self.assertEqual(response.status_code, 405)
        self.assertEqual(PredictionHistory.objects.filter(user=self.user_a).count(), 1)

    # Test 15: CSRF protection remains active
    def test_csrf_protection_remains_active(self):
        client_no_csrf = Client(enforce_csrf_checks=True)
        client_no_csrf.login(username='usera', password='password123')
        response = client_no_csrf.post(self.detect_url, {'dur': 0.1})
        self.assertEqual(response.status_code, 403)
