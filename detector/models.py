from django.db import models
from django.conf import settings


class PredictionHistory(models.Model):
    """
    Stores each intrusion detection prediction made by the system, linked to a User.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='prediction_histories',
        null=True,
        blank=True
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    source_ip = models.CharField(max_length=45, blank=True, default='N/A')
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField(default=0.0)
    attack_type = models.CharField(max_length=100, blank=True, default='N/A')
    is_threat = models.BooleanField(default=False)
    raw_input = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Prediction History'
        verbose_name_plural = 'Prediction Histories'

    def __str__(self):
        user_str = self.user.username if self.user else 'Anonymous'
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {user_str} - {self.prediction} (Threat: {self.is_threat})"
