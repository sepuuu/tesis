import multiprocessing as mp
from ultralytics import YOLO, YAML
import koila

def train_model():

    model = YOLO('modelos/yolov8x.pt')
    
    data = YAML('config.yaml') # Cargar archivo YAML
    dataset = data.train # Obtener dataset de entrenamiento
    
    wrapped_dataset = koila.Data(dataset) 

    results = model.train(data=wrapped_dataset, 
                          epochs=1, 
                          imgsz=640,  
                          device=0)

if __name__ == "__main__":

    mp.freeze_support()    

    process = mp.Process(target=train_model)
    process.start()
    process.join()