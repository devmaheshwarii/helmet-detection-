import cv2
import numpy as np

# Paths to the model files
weights_path = 'yolov4.weights'  # Use YOLOv4-tiny for faster processing
config_path = 'yolov4.cfg'
classes_path = 'coco.names'

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class names
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load video
cap = cv2.VideoCapture('helmetbike.mp4')

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for faster processing
    frame_count += 1
    if frame_count % 20 != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (608, 608))

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence >  0.5:  # Confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                label = str(classes[class_id])
                color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the frame
    from google.colab.patches import cv2_imshow
    cv2_imshow(frame)

    # Add a break condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
