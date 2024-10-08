import cv2 
from matplotlib import pyplot as plt


# cap = cv2.VideoCapture(0)

# Get a frame from the video source
# ret, frame = cap.read()
# print(ret)
# print(frame)

# cv2.imwrite("test.jpg", frame)

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.imshow(frame)
# plt.show()

# cap.release()

def take_photo(): 
    cap = cv2.VideoCapture(3)
    ret, frame = cap.read()
    cv2.imwrite('webcamphoto.jpg', frame)
    cap.release()
    
# take_photo()


# Connect to webcam
cap = cv2.VideoCapture(0)
# Loop through every frame until we close our webcam
while cap.isOpened(): 
    ret, frame = cap.read()
    
    # Show image 
    cv2.imshow('Webcam', frame)
    
    # Checks whether q has been hit and stops the loop
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Releases the webcam
cap.release()
# Closes the frame
cv2.destroyAllWindows()