## Ultralytics-yolo

github: [ultralytics/ultralytics: Ultralytics YOLO11 🚀](https://github.com/ultralytics/ultralytics)

docs: https://docs.ultralytics.com/

### install 

需要环境：pyton>3.8 pytorch>=1.8

```
pip install ultralytics 
```

### python 

``` python
from ultralytics import YOLO

# load a model
model = YOLO('yolo11n.pt')

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```



