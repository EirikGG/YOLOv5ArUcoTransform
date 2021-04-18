import cv2
import numpy as np

def transform_to_new_coord(point, pixel_coords, res=(100,100)):
    src = np.float32(pixel_coords)
    
    dst = np.float32([
        [0,0],
        [res[0]-1,0],
        [res[0]-1,res[1]-1],
        [0,res[1]-1]
    ])

    h, _ = cv2.findHomography(src, dst)
    
    new_p = np.dot((point[0], point[1], 1), h.T)

    return new_p[:-1]

