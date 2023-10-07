import cv2
import torch

# Carga el modelo YOLOv5 utilizando torch.hub
modelo = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)

# Configura la detección
umbral_confianza = 0.3  # Umbral de confianza para la detección
umbral_iou = 0.5   # Umbral de IoU para la supresión no máxima

# Abre el archivo de video
ruta_video = 'video2.mp4'  # Cambia esto al nombre de tu archivo .mp4
cap = cv2.VideoCapture(ruta_video)

# Configura la salida de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_salida = cv2.VideoWriter('video_salida2.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, cuadro = cap.read()
    if not ret:
        break

    # Realiza la detección de objetos
    resultados = modelo(cuadro)

    # Accede a las detecciones
    detecciones = resultados.pred[0]

    # Aplica los filtros de confianza e IoU
    mascara = detecciones[:, 4] > umbral_confianza
    detecciones = detecciones[mascara]

    # Dibuja los cuadros delimitadores y etiquetas en el cuadro
    for det in detecciones:
        xyxy = det[:4].cpu().numpy().astype(int)
        etiqueta = f'{resultados.names[int(det[5])]} {det[4]:.2f}'
        cv2.rectangle(cuadro, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(cuadro, etiqueta, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,), 2)

    # Muestra el cuadro en una ventana
    cv2.imshow('Video', cuadro)
    
    # Guarda el cuadro en el video resultante
    video_salida.write(cuadro)

    # Espera a que se pulse una tecla
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
video_salida.release()
cv2.destroyAllWindows()
