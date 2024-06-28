from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login, logout
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import ImageFeed, DetectedObject
from .forms import ImageFeedForm
from .utils import process_image_with_detr, process_image_with_ssd, get_plot
from django.views.decorators.http import require_POST
import matplotlib.pyplot as plt
import json
import io
import base64
def home(request):
    return render(request, 'object_detection/home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('object_detection:dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'object_detection/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('object_detection:dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'object_detection/login.html', {'form': form})

@login_required
def user_logout(request):
    logout(request)
    return redirect('object_detection:login')

@login_required
def dashboard(request):
    image_feeds = ImageFeed.objects.filter(user=request.user)
    return render(request, 'object_detection/dashboard.html', {'image_feeds': image_feeds})

@login_required
@require_POST
def process_image(request, feed_id):
    image_feed = get_object_or_404(ImageFeed, id=feed_id, user=request.user)
    data = json.loads(request.body)
    model = data.get('model', 'ssd')

    if model == 'detr':
        processed_image_url = process_image_with_detr(feed_id)
    elif model == 'ssd':
        processed_image_url = process_image_with_ssd(feed_id)
    else:
        return JsonResponse({"error": "Invalid model"}, status=400)

    if processed_image_url:
        detected_objects = list(image_feed.detected_objects.all().values('object_type', 'confidence'))
        graph = generate_graph(detected_objects)
        return JsonResponse({'processed_image_url': processed_image_url, 'detected_objects': detected_objects, 'graph': graph})
    else:
        return JsonResponse({'error': 'Processing failed'}, status=500)

def generate_graph(detected_objects):
    objects = [obj['object_type'] for obj in detected_objects]
    confidences = [obj['confidence'] for obj in detected_objects]

    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 5))
    plt.bar(objects, confidences, width=0.5, edgecolor="white", linewidth=0.7)
    plt.yscale('log')
    plt.title("Detected Objects")
    plt.xlabel('Objects')
    plt.ylabel('Confidence')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    return graph

@login_required
def add_image_feed(request):
    if request.method == 'POST':
        form = ImageFeedForm(request.POST, request.FILES)
        if form.is_valid():
            image_feed = form.save(commit=False)
            image_feed.user = request.user
            image_feed.save()
            return redirect('object_detection:dashboard')
    else:
        form = ImageFeedForm()
    return render(request, 'object_detection/add_image_feed.html', {'form': form})

@login_required
def delete_image(request, image_id):
    image = get_object_or_404(ImageFeed, id=image_id, user=request.user)
    image.delete()
    return redirect('object_detection:dashboard')

@login_required
def image_detect(request, pk):
    image_feed = get_object_or_404(ImageFeed, id=pk, user=request.user)
    detected_objects = DetectedObject.objects.filter(image_feed=pk)

    x = [obj.object_type for obj in detected_objects]
    y = [obj.confidence for obj in detected_objects]
    chart = get_plot(x, y, 'bar') if x and y else None

    context = {
        'image_feed': image_feed,
        'detected_objects': detected_objects,
        'chart': chart,
    }
    return render(request, "object_detection/image_detect.html", context=context)