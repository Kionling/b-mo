import cv2
# import tensorflow as tf

# # Load your pre-trained model (this is just a placeholder for the actual model loading process)
# # model = tf.saved_model.load('path_to_pretrained_model')

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Pre-process the frame for your model as necessary
#     # For example, resizing, normalization, etc.
#     input_frame = preprocess(frame)
    
#     # Object detection
#     detections = model(input_frame)
    
#     # Post-process detections and visualize them
#     # This typically includes drawing bounding boxes and labels on the frame
#     processed_frame = postprocess(frame, detections)
    
#     # Display the processed frame
#     cv2.imshow('Object Detection', processed_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# # Release the webcam and destroy all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
