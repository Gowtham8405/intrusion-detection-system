from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('analytics/', views.analytics, name='analytics'),
    path('detect/', views.detect, name='detect'),
    path('live-monitor/', views.live_monitor, name='live_monitor'),
    path('live-monitor/sample/', views.live_monitor_sample, name='live_monitor_sample'),
    path('history/', views.history, name='history'),
    path('history/<int:record_id>/delete/', views.delete_prediction, name='delete_prediction'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('about/', views.about, name='about'),
]
