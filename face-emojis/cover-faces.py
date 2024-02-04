#
# Author: Jeremy Pedersen (and ChatGPT)
# Updated: 2024-02-04
#
# Draw an emoji on top of all faces detected in a scene
import cv2
import dlib
from imutils import face_utils

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()

# Load your emoji images
emoji_img = cv2.imread('emoji.png', -1)  # Load with alpha channel

# Initialize video source, webcam in this case
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Determine height and width of frame
    height,width = frame.shape[:2]

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    # Loop over the face detections
    for face in faces:
        # Get the coordinates of the face
        (x, y, w, h) = face_utils.rect_to_bb(face)

        # Re-scale smiley to fit over entire face (and adjust (x,y) coordinates respectively)
        (x, y, w, h) = (x - int(w/2), y - int(w/2), w*2, h*2)

        # Resize emoji to the size of the face
        emoji_resized = cv2.resize(emoji_img, (w, h))
        
        # Create an overlay with the emoji
        for i in range(0, w):
            for j in range(0, h):
                if emoji_resized[j, i, 3] != 0:  # Check if it's not a transparent pixel

                    # Do not attempt to draw outside the frame
                    if y + j >= height or y + j < 0:
                        continue

                    if x + i >= width or x + i < 0:
                        continue

                    # DEBUG
                    # print(f'y+j: {y+j}')
                    # print(f'x+i: {x+i}')

                    frame[y + j, x + i] = emoji_resized[j, i, 0:3]

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
