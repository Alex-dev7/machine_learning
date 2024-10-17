import cv2
from matplotlib import pyplot as plt

# pip install opencv-contrib-python

# Initialize tracker
tracker = cv2.TrackerKCF_create()
init_box = None  # Assuming init_box is the bounding box of detected object
video = cv2.VideoCapture(0)

# Skip first N frames because of the camera fadein
n = 10  
for _ in range(n):
    _, _ = video.read()


# Read first frame
ok, frame = video.read()
if ok:
    init_box = cv2.selectROI(frame, False)  
    tracker.init(frame, init_box)

while True:
    ok, frame = video.read()
    if not ok:
        break
    
    ok, bbox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()