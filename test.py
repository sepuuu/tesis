import cv2
import torch

# Carga el modelo YOLOv5 usando torch.hub
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)

# Configura la detección
conf_thres = 0.5  # Umbral de confianza para la detección
iou_thres = 0.5   # Umbral de IoU para la supresión no máxima

# Abre el archivo de video
video_path = 'video2.mp4'  # Cambia esto al nombre de tu archivo .mp4
cap = cv2.VideoCapture(video_path)

# Configura la salida de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la detección de objetos
    results = model(frame)

    # Accede a las detecciones
    detections = results.pred[0]

    # Aplica los filtros de confianza e IoU
    mask = detections[:, 4] > conf_thres
    detections = detections[mask]

    # Dibuja los cuadros delimitadores y etiquetas en el frame
    for det in detections:
        xyxy = det[:4].cpu().numpy().astype(int)
        label = f'{results.names[int(det[5])]} {det[4]:.2f}'
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Guarda el frame en el video resultante
    output_video.write(frame)

cap.release()
output_video.release()
