from ultralytics import YOLO
from ultralytics.utils.loss import CustomLoss

model_cfg = "./yolov8n_modified.yaml"  # Replace with your actual path
model = YOLO(model_cfg)

# Set your custom loss function as the loss function for the YOLOv8 model
model.model.loss = CustomLoss(lambda1=1.5, lambda2=0.5, lambda3=2.0, alpha=1.0, beta=2.0, delta=1.0, gamma=1.5) 


# Train the model with your custom loss function
model.train(data='data.yaml', epochs=50)