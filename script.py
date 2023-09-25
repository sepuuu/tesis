import cv2
import numpy as np

# Carga el modelo YOLO
model = cv2.dnn.readNet("yolo.weights", "yolo.cfg")

# Carga el video
cap = cv2.VideoCapture("video.mp4")

# Inicializa un diccionario para almacenar las coordenadas de los jugadores
players = {}

while True:
    # Captura un cuadro del video
    ret, frame = cap.read()

    # Detecta jugadores en el cuadro
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    detections = model.forward()

    # Actualiza el diccionario de jugadores
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x = int(detections[0, 0, i, 3])
            y = int(detections[0, 0, i, 4])
            w = int(detections[0, 0, i, 5])
            h = int(detections[0, 0, i, 6])

            # Crea un objeto jugador
            player = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }

            # Agrega el objeto jugador al diccionario
            players[i] = player

    # Dibuja los jugadores detectados
    for player in players.values():
        x = player["x"]
        y = player["y"]
        w = player["w"]
        h = player["h"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Muestra el cuadro
    cv2.imshow("Video", frame)

    # Termina el programa si se presiona la tecla q
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Cierra el video
cap.release()

# Destruye todas las ventanas abiertas
cv2.destroyAllWindows()
