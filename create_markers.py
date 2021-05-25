import cv2

import numpy as np

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
 
# Generate the marker
ns = (0, 1, 2, 3)
for n in ns:
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, n, 200, markerImage, 1)
    
    cv2.imwrite(f'marker{n}.png', markerImage)