from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2
import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from .models import DetectedObject, ImageFeed
import matplotlib.pyplot as plt
import io
import base64

DETR_MODEL_PATH = 'facebook/detr-resnet-50'
processor = DetrImageProcessor.from_pretrained(DETR_MODEL_PATH)
model = DetrForObjectDetection.from_pretrained(DETR_MODEL_PATH)

VOC_LABELS = [
    "фон", "самолет", "велосипед", "птица", "лодка", "бутылка",
    "автобус", "автомобиль", "кот", "стул", "корова", "стол",
    "собака", "лошадь", "мотоцикл", "человек", "цветок в горшке",
    "овца", "диван", "поезд", "телевизор"
]

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def detect_objects_detr(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    return results

def detect_objects_ssd(image_path):
    model_path = 'object_detection/mobilenet_iter_73000.caffemodel'
    config_path = 'object_detection/mobilenet_ssd_deploy.prototxt'
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение")
        return []

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            class_label = VOC_LABELS[class_id]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                'class_label': class_label,
                'confidence': confidence,
                'box': (startX, startY, endX, endY)
            })
    return results

def process_image_with_detr(image_feed_id):
    try:
        image_feed = ImageFeed.objects.get(id=image_feed_id)
        image_path = image_feed.image.path

        image = Image.open(image_path)

        results = detect_objects_detr(image)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

            DetectedObject.objects.create(
                image_feed=image_feed,
                object_type=model.config.id2label[label.item()],
                location=f"{box[0]},{box[1]},{box[2]},{box[3]}",
                confidence=float(score.item())
            )

            image = cv2.rectangle(np.array(image), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            label = f'{model.config.id2label[label.item()]}: {round(score.item(), 2)}'
            image = cv2.putText(np.array(image), label, (int(box[0]) + 5, int(box[1]) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        result, encoded_img = cv2.imencode('.jpg', image)
        if result:
            content = ContentFile(encoded_img.tobytes(), f'processed_{image_feed.image.name}')
            image_feed.processed_image.save(content.name, content, save=True)

        return image_feed.processed_image.url

    except ImageFeed.DoesNotExist:
        print("ImageFeed не найден.")
        return False

def process_image_with_ssd(image_feed_id):
    try:
        image_feed = ImageFeed.objects.get(id=image_feed_id)
        image_path = image_feed.image.path

        img = cv2.imread(image_path)
        if img is None:
            print("Не удалось загрузить изображение")
            return False

        results_ssd = detect_objects_ssd(image_path)

        for result in results_ssd:
            class_label = result['class_label']
            confidence = result['confidence']
            startX, startY, endX, endY = result['box']

            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            label = f"{class_label}: {confidence:.2f}"
            cv2.putText(img, label, (startX + 5, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            DetectedObject.objects.create(
                image_feed=image_feed,
                object_type=class_label,
                location=f"{startX},{startY},{endX},{endY}",
                confidence=float(confidence)
            )

        result, encoded_img = cv2.imencode('.jpg', img)
        if result:
            content = ContentFile(encoded_img.tobytes(), f'processed_{image_feed.image.name}')

            # Удаляем старое обработанное изображение, если оно существует
            if image_feed.processed_image:
                image_feed.processed_image.delete()

            image_feed.processed_image.save(content.name, content, save=True)

        return image_feed.processed_image.url

    except ImageFeed.DoesNotExist:
        print("ImageFeed не найден.")
        return False

def get_graph():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(x, y, type_graph):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 'large',
            }
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    plt.rc('font', size=20)
    plt.switch_backend('AGG')
    plt.figure(figsize=(15, 5))
    plt.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
    plt.yscale('log')
    plt.title("График", fontdict=font)
    plt.xlabel('Объекты', fontdict=font)
    plt.ylabel('Вероятность', fontdict=font)
    plt.tight_layout()
    plt.savefig(type_graph + '.png')

    return get_graph()

def get_plot_stat(x, y, type_graph):
    plt.switch_backend('AGG')
    plt.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

    plt.yscale('log')
    plt.title("График")
    plt.xlabel('Метод', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.savefig(type_graph + '.png')

    return get_graph()