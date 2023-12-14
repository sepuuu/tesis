import cv2
from ultralytics import YOLO
import threading

def process_video_in_thread(video_path, output_path, model, file_index):
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object for AVI output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, conf=0.1, persist=True)
            result = results[0]

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                conf = results[0].boxes.conf.cpu()

                # Accede al diccionario de nombres
                class_names_dict = result.names

                for box, id, class_id, conf_value in zip(boxes, ids, class_ids, conf):
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
                        1,
                    )

            output_video.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

# Load the YOLOv8 models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')

# Define video paths for processing
video1_path = "videos/entrada/albo.mp4"
video2_path = "videos/entrada/video2.mp4"

# Create threads for processing videos
thread1 = threading.Thread(target=process_video_in_thread, args=(video1_path, "videos/salida/output_video1.avi", model1, 1))
thread2 = threading.Thread(target=process_video_in_thread, args=(video2_path, "videos/salida/output_video2.avi", model2, 2))

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()
