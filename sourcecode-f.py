import cv2
from google.colab.patches import cv2_imshow
from datetime import datetime
import pytz

cascade_src = 'bike.xml'
video_src = 'helmetbike.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

ist = pytz.timezone('Asia/Kolkata')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect objects
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(cars) == 0:  # No helmet detected
        # Add label and timestamp to the frame
        cv2.putText(frame, 'Helmet Not Present', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        timestamp = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2_imshow(frame)
        break  # Stop after displaying the first frame without a helmet

cap.release()
cv2.destroyAllWindows()
