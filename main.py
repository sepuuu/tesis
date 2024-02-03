

import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('modelos/bestv3.pt')

# Open the video file
video_path = "videos/entrada/clip_dron.mp4"
cap = cv2.VideoCapture(video_path)

# Load the 2D field image
field_img = cv2.imread('cancha.png') 
field_img_copy = field_img.copy()  # Copia de la imagen para evitar rastro de cuadros

# Puntos para la homografía
dst_pts = np.array([ ##mapa 2d
  [76, 70], 
  [958, 74],
  [1820, 76],
  [80, 471],
  [80, 607],
  [345, 537],
  [85, 1012],
  [958, 544],   
  [960, 1007],
  [1817, 474],
  [1817, 608],
  [1558, 544], 
  [1814, 1002]   
])

src_pts = np.array([  #video
  [120, 239], 
  [975, 21],
  [1852, 9],
  [120, 434],
  [122, 566],
  [381, 500],
  [125, 964],
  [981, 490],   
  [987, 964],
  [1859, 416],
  [1862, 553],
  [1591, 488], 
  [1869, 953]     
])
output_txt_file = 'posiciones_jugadores.txt'
# Calcular homografía
h, _ = cv2.findHomography(src_pts, dst_pts)

# Define the codec and create VideoWriter object for AVI output
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_video = cv2.VideoWriter('videos/salida/output_video6.avi', fourcc, 30.0, (854*2, 480))
posiciones_jugador = []

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.1, persist=True, iou=0.2)
        result = results[0]

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            conf = results[0].boxes.conf.cpu() 
            class_names_dict = result.names

            # Proyecta las bounding boxes usando la homografía
            warped_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                verts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                warped_verts = cv2.perspectiveTransform(verts.reshape(-1, 1, 2), h)

                min_x = warped_verts[:,:,0].min()
                max_x = warped_verts[:,:,0].max()
                min_y = warped_verts[:,:,1].min()
                max_y = warped_verts[:,:,1].max()
                warped_box = [min_x, min_y, max_x, max_y]
                warped_boxes.append(warped_box)

                print(boxes[2])
                
                
        if len(warped_boxes) > 1:
            pos_jugador = warped_boxes[6]
            posiciones_jugador.append(pos_jugador)
            print(pos_jugador)
        else:
            print("No hay suficientes jugadores detectados en este cuadro.")                   
            # Dibuja los bounding boxes transformados en la imagen del campo 2D
# Dibuja los puntos en el centro de masa de los bounding boxes transformados en la imagen del campo 2D
        field_img = field_img_copy.copy()  # Restaura la copia original antes de dibujar
        for warped_box, id, class_id, conf_value in zip(warped_boxes, ids, class_ids, conf):
            x1, y1, x2, y2 = map(int, warped_box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(field_img, (center_x, center_y), 15, (255, 255, 255), -1)

            
            # Resto del código para mostrar la información y el texto en el cuadro...


            for box, id, class_id, conf_value in zip(boxes, ids, class_ids,conf):
                # Accede al nombre de clase correspondiente utilizando la class_id
                if class_id in class_names_dict:
                    class_name = class_names_dict[class_id]
                else:
                    class_name = 'Desconocido'

                # Construye el texto para mostrar en el cuadro
                text = f"{id} {class_name} {conf_value:.2f}"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,)

        # Escala las imágenes para que quepan en una pantalla de 1920x1080
        frame_resized = cv2.resize(frame, (854, 480))
        field_img_resized = cv2.resize(field_img, (854, 480))

        # Muestra el video con el tracking y la imagen del campo 2D al lado
        concatenated_frame = np.concatenate((frame_resized, field_img_resized), axis=1)
        output_video.write(concatenated_frame)
        cv2.imshow("YOLOv8 Tracking & Homografía", concatenated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

np.savetxt(output_txt_file, posiciones_jugador, fmt='%d')
print(posiciones_jugador)
cap.release()
output_video.release()
cv2.destroyAllWindows()
