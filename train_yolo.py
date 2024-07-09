from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-cls.pt")  # load a pretrained model (recommended for training)

import os
os.environ["WANDB_DISABLED"] = 'true'
model.train(data="multimodal_data_v2/", epochs=20)  # train the model