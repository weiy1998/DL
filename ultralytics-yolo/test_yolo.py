from ultralytics import YOLO
import cv2

def inference_single_image():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")
    # Perform object detection on an image using the model
    results = model("bus.jpg")
    # 
    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        result.show()
        result.save(filename='result.jpg')
    

def inference_video():
    # load model
    model = YOLO('yolo11n.pt')
    # create capture object
    cap = cv2.VideoCapture(0)  # 使用本地相机，如果使用视频文件，则使用 cv2.VideoCapture('/path/to/video.mp4')

    if not cap.isOpened():
        print("Error: Could not open video.")
        return 
    
    while True:
        # get frane
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return 
        
        # yolo model inference 
        results = model(frame)
        
        # view results
        annotated_frame = results[0].plot()
        cv2.imshow('yolo11n inference', annotated_frame)

        # 退出窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 销毁窗口，释放资源
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    # inference_single_image()
    inference_video()