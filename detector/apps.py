from django.apps import AppConfig


class DetectorConfig(AppConfig):
    """Configuration for the detector application."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detector'
    verbose_name = 'Intrusion Detection System'
