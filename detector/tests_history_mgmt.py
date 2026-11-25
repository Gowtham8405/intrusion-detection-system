"""
detector/tests_history_mgmt.py

Automated Prediction History Management Tests for Step 9.
Tests individual prediction delete, clear all history, CSRF protection, and cross-user authorization safety.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from detector.models import PredictionHistory


class HistoryManagementAndAuthorizationTests(TestCase):

    def setUp(self):
        self.client_a = Client()
        self.client_b = Client()

        self.user_a = User.objects.create_user(username='history_user_a', password='Password123!')
        self.user_b = User.objects.create_user(username='history_user_b', password='Password123!')

        self.client_a.login(username='history_user_a', password='Password123!')
        self.client_b.login(username='history_user_b', password='Password123!')

        # Create initial history records
        self.rec_a1 = PredictionHistory.objects.create(user=self.user_a, prediction="NORMAL TRAFFIC", confidence=98.5, is_threat=False)
        self.rec_a2 = PredictionHistory.objects.create(user=self.user_a, prediction="ATTACK DETECTED", confidence=95.2, is_threat=True)
        self.rec_b1 = PredictionHistory.objects.create(user=self.user_b, prediction="ATTACK DETECTED", confidence=99.1, is_threat=True)

    def test_01_authenticated_user_can_view_own_history(self):
        """TEST 1: Authenticated user can view own history records."""
        response = self.client_a.get(reverse('detector:history'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "NORMAL TRAFFIC")
        self.assertContains(response, "ATTACK DETECTED")

    def test_02_user_a_sees_only_user_a_records(self):
        """TEST 2: User A sees only User A records."""
        response = self.client_a.get(reverse('detector:history'))
        records = list(response.context['records'])
        self.assertIn(self.rec_a1, records)
        self.assertIn(self.rec_a2, records)
        self.assertNotIn(self.rec_b1, records)

    def test_03_user_b_sees_only_user_b_records(self):
        """TEST 3: User B sees only User B records."""
        response = self.client_b.get(reverse('detector:history'))
        records = list(response.context['records'])
        self.assertIn(self.rec_b1, records)
        self.assertNotIn(self.rec_a1, records)
        self.assertNotIn(self.rec_a2, records)

    def test_04_individual_delete_works_for_own_record(self):
        """TEST 4: Individual delete via POST deletes user's own record."""
        response = self.client_a.post(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_a1.id}))
        self.assertRedirects(response, reverse('detector:history'))
        self.assertFalse(PredictionHistory.objects.filter(id=self.rec_a1.id).exists())

    def test_05_deleted_record_no_longer_in_history(self):
        """TEST 5: Deleted record no longer appears in user's history page context."""
        self.client_a.post(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_a1.id}))
        response = self.client_a.get(reverse('detector:history'))
        records = list(response.context['records'])
        self.assertNotIn(self.rec_a1, records)

    def test_06_user_a_cannot_delete_user_b_record(self):
        """TEST 6: User A trying to delete User B's record returns 404 Not Found."""
        response = self.client_a.post(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_b1.id}))
        self.assertEqual(response.status_code, 404)

    def test_07_user_b_record_remains_intact_after_attempted_cross_delete(self):
        """TEST 7: User B's record remains intact after User A attempts deletion."""
        self.client_a.post(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_b1.id}))
        self.assertTrue(PredictionHistory.objects.filter(id=self.rec_b1.id).exists())

    def test_08_clear_history_deletes_all_current_user_records(self):
        """TEST 8: Clear history POST deletes all records of the currently logged-in user."""
        response = self.client_a.post(reverse('detector:clear_history'))
        self.assertRedirects(response, reverse('detector:history'))
        self.assertEqual(PredictionHistory.objects.filter(user=self.user_a).count(), 0)

    def test_09_clear_history_does_not_delete_another_user_records(self):
        """TEST 9: Clear history by User A does NOT delete User B's records."""
        self.client_a.post(reverse('detector:clear_history'))
        self.assertEqual(PredictionHistory.objects.filter(user=self.user_b).count(), 1)
        self.assertTrue(PredictionHistory.objects.filter(id=self.rec_b1.id).exists())

    def test_10_individual_delete_requires_post(self):
        """TEST 10: Individual delete via GET returns 405 Method Not Allowed."""
        response = self.client_a.get(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_a1.id}))
        self.assertEqual(response.status_code, 405)
        self.assertTrue(PredictionHistory.objects.filter(id=self.rec_a1.id).exists())

    def test_11_clear_history_requires_post(self):
        """TEST 11: Clear history via GET returns 405 Method Not Allowed."""
        response = self.client_a.get(reverse('detector:clear_history'))
        self.assertEqual(response.status_code, 405)
        self.assertEqual(PredictionHistory.objects.filter(user=self.user_a).count(), 2)

    def test_12_individual_delete_csrf_enforcement(self):
        """TEST 12: Individual delete form renders with CSRF token."""
        response = self.client_a.get(reverse('detector:history'))
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_13_clear_history_csrf_enforcement(self):
        """TEST 13: Clear history form renders with CSRF token."""
        response = self.client_a.get(reverse('detector:history'))
        self.assertContains(response, "csrfmiddlewaretoken")

    def test_14_unauthenticated_user_blocked_from_delete(self):
        """TEST 14: Unauthenticated user cannot access delete endpoint."""
        unauth_client = Client()
        response = unauth_client.post(reverse('detector:delete_prediction', kwargs={'record_id': self.rec_a1.id}))
        self.assertRedirects(response, f"{reverse('detector:login')}?next={reverse('detector:delete_prediction', kwargs={'record_id': self.rec_a1.id})}")

    def test_15_unauthenticated_user_blocked_from_clear(self):
        """TEST 15: Unauthenticated user cannot access clear history endpoint."""
        unauth_client = Client()
        response = unauth_client.post(reverse('detector:clear_history'))
        self.assertRedirects(response, f"{reverse('detector:login')}?next={reverse('detector:clear_history')}")

    def test_16_empty_history_state_displays_friendly_message(self):
        """TEST 16: Empty history state displays friendly message and CTA link when user has no records."""
        self.client_a.post(reverse('detector:clear_history'))
        response = self.client_a.get(reverse('detector:history'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No Prediction History Yet")
        self.assertContains(response, "Run Your First Detection")
