from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    """Admin interface for viewing prediction history records."""
    list_display = ('timestamp', 'user', 'prediction', 'confidence', 'attack_type', 'is_threat')
    list_filter = ('is_threat', 'prediction', 'user')
    search_fields = ('user__username', 'source_ip', 'prediction', 'attack_type')
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)
