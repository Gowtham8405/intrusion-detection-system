from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseNotAllowed, JsonResponse
from django.db import models
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from .models import PredictionHistory
from .ml.predictor import IntrusionPredictor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from .live_monitor import execute_live_prediction

# Sample Presets from Validation Dataset for quick manual UI testing
PRESET_NORMAL = {
    'dur': 0.000011, 'proto': 'udp', 'service': '-', 'state': 'INT',
    'spkts': 2, 'dpkts': 0, 'sbytes': 104, 'dbytes': 0, 'rate': 90909.09,
    'sttl': 254, 'dttl': 0, 'sload': 37818180.0, 'dload': 0.0, 'sloss': 0, 'dloss': 0,
    'sinpkt': 0.011, 'dinpkt': 0.0, 'sjit': 0.0, 'djit': 0.0, 'swin': 0,
    'stcpb': 0, 'dtcpb': 0, 'dwin': 0, 'tcprtt': 0.0, 'synack': 0.0, 'ackdat': 0.0,
    'smean': 52, 'dmean': 0, 'trans_depth': 0, 'response_body_len': 0,
    'ct_srv_src': 2, 'ct_state_ttl': 2, 'ct_dst_ltm': 1, 'ct_src_dport_ltm': 1,
    'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 2, 'is_ftp_login': 0, 'ct_ftp_cmd': 0,
    'ct_flw_http_mthd': 0, 'ct_src_ltm': 1, 'ct_srv_dst': 2, 'is_sm_ips_ports': 0
}

PRESET_ATTACK = {
    'dur': 0.924876, 'proto': 'tcp', 'service': 'http', 'state': 'FIN',
    'spkts': 10, 'dpkts': 8, 'sbytes': 800, 'dbytes': 3534, 'rate': 18.381,
    'sttl': 62, 'dttl': 252, 'sload': 6919.85, 'dload': 27495.58, 'sloss': 2, 'dloss': 2,
    'sinpkt': 102.76, 'dinpkt': 118.23, 'sjit': 4229.83, 'djit': 284.14, 'swin': 255,
    'stcpb': 2450893322, 'dtcpb': 2447990422, 'dwin': 255, 'tcprtt': 0.00078,
    'synack': 0.00053, 'ackdat': 0.00025, 'smean': 80, 'dmean': 442,
    'trans_depth': 1, 'response_body_len': 268, 'ct_srv_src': 1, 'ct_state_ttl': 1,
    'ct_dst_ltm': 1, 'ct_src_dport_ltm': 1, 'ct_dst_sport_ltm': 1, 'ct_dst_src_ltm': 1,
    'is_ftp_login': 0, 'ct_ftp_cmd': 0, 'ct_flw_http_mthd': 1, 'ct_src_ltm': 1,
    'ct_srv_dst': 1, 'is_sm_ips_ports': 0
}


def home(request):
    """Homepage — Project overview with navigation to all sections."""
    context = {
        'title': 'AI-Based Intrusion Detection System',
    }
    return render(request, 'detector/home.html', context)


def register_view(request):
    """User Registration View."""
    if request.user.is_authenticated:
        return redirect('detector:detect')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        # Validations
        if not username or not password:
            messages.error(request, "Username and Password are required.")
        elif password != confirm_password:
            messages.error(request, "Passwords do not match.")
        elif len(password) < 6:
            messages.error(request, "Password must be at least 6 characters long.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists. Please choose a different username.")
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            messages.success(request, f"Account created for '{username}'! Please log in.")
            return redirect('detector:login')

    return render(request, 'registration/register.html', {'title': 'Register'})


def login_view(request):
    """User Login View."""
    if request.user.is_authenticated:
        return redirect('detector:detect')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')

        if not username or not password:
            messages.error(request, "Please enter both username and password.")
        else:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                next_url = request.GET.get('next')
                if next_url:
                    return redirect(next_url)
                return redirect('detector:detect')
            else:
                messages.error(request, "Invalid username or password.")

    return render(request, 'registration/login.html', {'title': 'Login'})


def logout_view(request):
    """User Logout View."""
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('detector:login')


@login_required(login_url='detector:login')
def dashboard(request):
    """Dashboard — User-specific intrusion detection activity summary."""
    user_predictions = PredictionHistory.objects.filter(user=request.user)
    total_predictions = user_predictions.count()
    attacks_count = user_predictions.filter(is_threat=True).count()
    normal_count = total_predictions - attacks_count
    
    if total_predictions > 0:
        threat_percentage = round((attacks_count / total_predictions) * 100, 1)
        normal_percentage = round((normal_count / total_predictions) * 100, 1)
    else:
        threat_percentage = 0.0
        normal_percentage = 0.0

    recent_predictions = user_predictions.order_by('-timestamp', '-id')[:5]

    context = {
        'title': 'User Dashboard',
        'total_predictions': total_predictions,
        'attacks_count': attacks_count,
        'normal_count': normal_count,
        'threat_percentage': threat_percentage,
        'normal_percentage': normal_percentage,
        'recent_predictions': recent_predictions,
    }
    return render(request, 'detector/dashboard.html', context)


@login_required(login_url='detector:login')
def analytics(request):
    """
    Prediction Analytics & Reports View.
    Displays summary stats, attack type breakdown, time-based recent activity,
    latest 10 predictions, and average confidence for request.user ONLY.
    """
    user_predictions = PredictionHistory.objects.filter(user=request.user)
    total_predictions = user_predictions.count()
    attacks_count = user_predictions.filter(is_threat=True).count()
    normal_count = total_predictions - attacks_count

    if total_predictions > 0:
        threat_percentage = round((attacks_count / total_predictions) * 100, 1)
        normal_percentage = round((normal_count / total_predictions) * 100, 1)
        avg_confidence = round(user_predictions.aggregate(Avg('confidence'))['confidence__avg'] or 0.0, 1)
    else:
        threat_percentage = 0.0
        normal_percentage = 0.0
        avg_confidence = 0.0

    # 1. Attack Type Breakdown (grouped by attack_type, count descending)
    attack_type_qs = (
        user_predictions
        .values('attack_type')
        .annotate(count=Count('id'))
        .order_by('-count')
    )
    
    attack_type_breakdown = []
    for item in attack_type_qs:
        raw_type = item['attack_type']
        display_type = raw_type.strip() if raw_type and raw_type.strip() else 'N/A'
        cnt = item['count']
        pct = round((cnt / total_predictions) * 100, 1) if total_predictions > 0 else 0.0
        attack_type_breakdown.append({
            'attack_type': display_type,
            'count': cnt,
            'percentage': pct
        })

    # 2. Time-Based Activity (Grouped by Date for user's predictions)
    daily_qs = (
        user_predictions
        .annotate(date=TruncDate('timestamp'))
        .values('date')
        .annotate(
            total=Count('id'),
            normal=Count('id', filter=Q(is_threat=False)),
            attacks=Count('id', filter=Q(is_threat=True))
        )
        .order_by('-date')[:7]
    )

    # 3. Latest 10 Predictions
    latest_10_predictions = user_predictions.order_by('-timestamp', '-id')[:10]

    context = {
        'title': 'Prediction Analytics & Reports',
        'total_predictions': total_predictions,
        'normal_count': normal_count,
        'attacks_count': attacks_count,
        'threat_percentage': threat_percentage,
        'normal_percentage': normal_percentage,
        'avg_confidence': avg_confidence,
        'attack_type_breakdown': attack_type_breakdown,
        'daily_activity': daily_qs,
        'latest_10_predictions': latest_10_predictions,
    }
    return render(request, 'detector/analytics.html', context)


@login_required(login_url='detector:login')
def detect(request):
    """
    Intrusion Detection View (Protected).
    Accepts network traffic feature input and classifies it as Normal or Attack using Random Forest.
    Saves prediction linked to request.user.
    """
    predictor = IntrusionPredictor()
    result = None
    form_data = PRESET_NORMAL.copy()  # Default values

    preset = request.GET.get('preset')
    if preset == 'normal':
        form_data = PRESET_NORMAL.copy()
        messages.info(request, "Loaded sample NORMAL network traffic features into form.")
    elif preset == 'attack':
        form_data = PRESET_ATTACK.copy()
        messages.info(request, "Loaded sample ATTACK network traffic features into form.")

    if request.method == 'POST':
        # Extract inputs from POST (excluding forbidden id, label, attack_cat)
        user_input = {}
        for col in FEATURE_COLUMNS:
            if col in request.POST:
                user_input[col] = request.POST.get(col)
                form_data[col] = request.POST.get(col)

        # Run Prediction
        prediction_res = predictor.predict_sample(user_input)

        if prediction_res['success']:
            result = prediction_res
            # Persist to database history linked to authenticated user
            PredictionHistory.objects.create(
                user=request.user,
                source_ip=request.POST.get('source_ip', '192.168.1.105'),
                prediction=prediction_res['prediction'],
                confidence=prediction_res['confidence'],
                attack_type='Machine Learning Classified' if prediction_res['is_attack'] else 'None',
                is_threat=prediction_res['is_attack'],
                raw_input=str(user_input)
            )
            if prediction_res['is_attack']:
                messages.warning(request, "⚠️ Malicious Network Intrusion Detected!")
            else:
                messages.success(request, "✅ Network Traffic Classified as Normal.")
        else:
            for err in prediction_res['errors']:
                messages.error(request, f"Validation Error: {err}")

    context = {
        'title': 'Intrusion Detection',
        'form_data': form_data,
        'result': result,
        'proto_choices': ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'unas'],
        'service_choices': ['-', 'http', 'dns', 'smtp', 'ftp', 'ftp-data', 'pop3', 'ssh', 'ssl', 'snmp'],
        'state_choices': ['FIN', 'INT', 'CON', 'REQ', 'ACC', 'RST', 'CLO']
    }
    return render(request, 'detector/detect.html', context)


@login_required(login_url='detector:login')
def history(request):
    """Prediction History (Protected & User-Specific)."""
    records = PredictionHistory.objects.filter(user=request.user).order_by('-timestamp')
    context = {
        'title': 'Prediction History',
        'records': records
    }
    return render(request, 'detector/history.html', context)


@login_required(login_url='detector:login')
def delete_prediction(request, record_id):
    """
    Delete an individual prediction record.
    Requires POST request and verifies ownership via get_object_or_404(user=request.user).
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    record = get_object_or_404(PredictionHistory, id=record_id, user=request.user)
    record.delete()
    messages.success(request, "Prediction record deleted successfully.")
    return redirect('detector:history')


@login_required(login_url='detector:login')
def clear_history(request):
    """
    Clear all prediction history records for the authenticated user.
    Requires POST request. Only deletes records belonging to request.user.
    """
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    deleted_count, _ = PredictionHistory.objects.filter(user=request.user).delete()
    messages.success(request, "Your prediction history has been cleared.")
    return redirect('detector:history')


def about(request):
    """About — Project information and methodology."""
    context = {
        'title': 'About',
    }
    return render(request, 'detector/about.html', context)


@login_required(login_url='detector:login')
def live_monitor(request):
    """
    Live Monitoring Demonstration View (Protected).
    Provides real-time local network/traffic monitoring interface,
    allowing users to toggle monitoring and evaluate local traffic against the ML model.
    """
    user_predictions = PredictionHistory.objects.filter(user=request.user)
    recent_events = user_predictions.order_by('-timestamp', '-id')[:10]
    total_count = user_predictions.count()
    attacks_count = user_predictions.filter(is_threat=True).count()

    context = {
        'title': 'Live Monitoring Demonstration',
        'recent_events': recent_events,
        'total_count': total_count,
        'attacks_count': attacks_count,
    }
    return render(request, 'detector/live_monitor.html', context)


@login_required(login_url='detector:login')
def live_monitor_sample(request):
    """
    AJAX Endpoint for Live Traffic Evaluation.
    Evaluates current local traffic activity via feature adapter and Random Forest ML model.
    """
    mode = request.GET.get('mode', 'normal')
    if request.method == 'POST':
        mode = request.POST.get('mode', mode)

    result = execute_live_prediction(request, request.user, mode=mode)
    return JsonResponse(result)

