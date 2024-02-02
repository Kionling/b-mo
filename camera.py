import cv2
import numpy as np

# Configuration and initialization
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
np.random.seed(543210)  # Seed for random colors
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the model
net = cv2.dnn.readNet(prototxt_path, model_path)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break  # If no frame is captured or returned, exit the loop

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            x_left = int(detected_objects[0, 0, i, 3] * width)
            y_top = int(detected_objects[0, 0, i, 4] * height)
            x_right = int(detected_objects[0, 0, i, 5] * width)
            y_bottom = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f'{classes[class_index]}: {confidence:.2f}%'
            cv2.rectangle(image, (x_left, y_top), (x_right, y_bottom), colors[class_index], 2)
            cv2.putText(image, prediction_text, (x_left, y_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_index], 2)

    cv2.imshow('Detected Objects', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
