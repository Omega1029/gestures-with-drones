import cv2

# Load the pre-trained Haar Cascade face detection model
# You can download the XML file from OpenCV's GitHub repository or use the built-in one.
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Capture video from webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar Cascades work on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    # The scaleFactor compensates for the faces being different sizes in the image.
    # minNeighbors helps to reduce false positives.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
