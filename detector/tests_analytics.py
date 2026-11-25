from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory
from django.utils import timezone
import datetime


class AnalyticsViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user_a = User.objects.create_user(username='usera', password='password123')
        self.user_b = User.objects.create_user(username='userb', password='password123')
        self.analytics_url = reverse('detector:analytics')
        self.history_url = reverse('detector:history')
        self.login_url = reverse('detector:login')

    # Test 1: Analytics requires authentication
    def test_analytics_requires_authentication(self):
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn(self.login_url, response.url)
        self.assertIn('next=/analytics/', response.url)

    # Test 2: Authenticated user can access analytics
    def test_authenticated_user_can_access_analytics(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'detector/analytics.html')

    # Test 3: Correct total prediction count
    def test_correct_total_prediction_count(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='DoS')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['total_predictions'], 2)

    # Test 4: Correct normal prediction count
    def test_correct_normal_prediction_count(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=92.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='Exploits')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['normal_count'], 2)

    # Test 5: Correct attack count
    def test_correct_attack_count(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=88.0, is_threat=True, attack_type='Generic')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['attacks_count'], 2)

    # Test 6: Correct threat percentage
    def test_correct_threat_percentage(self):
        # 1 Attack, 3 Normal -> 25% Threat Rate
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=96.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=94.0, is_threat=False, attack_type='None')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['threat_percentage'], 25.0)

    # Test 7: Zero predictions handled safely
    def test_zero_predictions_handled_safely(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_predictions'], 0)
        self.assertEqual(response.context['threat_percentage'], 0.0)
        self.assertEqual(response.context['avg_confidence'], 0.0)

    # Test 8: User-specific data isolation
    def test_user_specific_data_isolation(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_b, prediction='Attack', confidence=99.0, is_threat=True, attack_type='Fuzzers')
        
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['total_predictions'], 1)
        self.assertEqual(response.context['attacks_count'], 0)

    # Test 9: User A cannot see User B attack type statistics
    def test_user_a_cannot_see_user_b_attack_types(self):
        PredictionHistory.objects.create(user=self.user_b, prediction='Attack', confidence=99.0, is_threat=True, attack_type='Reconnaissance')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        attack_types = [item['attack_type'] for item in response.context['attack_type_breakdown']]
        self.assertNotIn('Reconnaissance', attack_types)

    # Test 10: User B cannot see User A analytics
    def test_user_b_cannot_see_user_a_analytics(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=91.0, is_threat=True, attack_type='Analysis')
        self.client.login(username='userb', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['total_predictions'], 0)
        self.assertContains(response, "No Analytics Data Yet")

    # Test 11: Attack type breakdown counts are correct
    def test_attack_type_breakdown_counts(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=92.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=85.0, is_threat=True, attack_type='Generic')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        breakdown = {item['attack_type']: item['count'] for item in response.context['attack_type_breakdown']}
        self.assertEqual(breakdown.get('DoS'), 2)
        self.assertEqual(breakdown.get('Generic'), 1)

    # Test 12: Attack types are ordered by count descending
    def test_attack_types_ordered_by_count_descending(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=90.0, is_threat=True, attack_type='Generic')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=92.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=92.0, is_threat=True, attack_type='DoS')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=92.0, is_threat=True, attack_type='DoS')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        breakdown = response.context['attack_type_breakdown']
        self.assertEqual(breakdown[0]['attack_type'], 'DoS')
        self.assertEqual(breakdown[0]['count'], 3)
        self.assertEqual(breakdown[1]['attack_type'], 'Generic')
        self.assertEqual(breakdown[1]['count'], 1)

    # Test 13: Recent activity is limited to the intended recent window
    def test_recent_activity_window(self):
        # Create predictions over multiple dates
        now = timezone.now()
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertLessEqual(len(response.context['daily_activity']), 7)

    # Test 14: Latest activity is limited to 10 records
    def test_latest_activity_limited_to_10(self):
        for i in range(15):
            PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=90.0 + i, is_threat=False, attack_type='None')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(len(response.context['latest_10_predictions']), 10)

    # Test 15: Empty analytics state renders correctly
    def test_empty_analytics_state_renders_correctly(self):
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertContains(response, "No Analytics Data Yet")
        self.assertContains(response, "Run Your First Detection")

    # Test 16: Existing history link is present
    def test_existing_history_link_present(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertContains(response, self.history_url)

    # Test 17: Analytics page does not modify prediction records
    def test_analytics_page_does_not_modify_records(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=95.0, is_threat=False, attack_type='None')
        initial_count = PredictionHistory.objects.count()
        self.client.login(username='usera', password='password123')
        self.client.get(self.analytics_url)
        final_count = PredictionHistory.objects.count()
        self.assertEqual(initial_count, final_count)

    # Test 18: Average confidence is calculated safely
    def test_average_confidence_calculated_safely(self):
        PredictionHistory.objects.create(user=self.user_a, prediction='Normal', confidence=90.0, is_threat=False, attack_type='None')
        PredictionHistory.objects.create(user=self.user_a, prediction='Attack', confidence=100.0, is_threat=True, attack_type='DoS')
        self.client.login(username='usera', password='password123')
        response = self.client.get(self.analytics_url)
        self.assertEqual(response.context['avg_confidence'], 95.0)
