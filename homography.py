import cv2
import numpy as np

def transform_to_new_coord(point, pixel_coords, res=(100,100)):
    '''Transforms a point to a new coodinate system based on
    pixel coordinates, which are assumed to be the corners of a
    square'''
    src = np.float32(pixel_coords)                  # Source square
    
    dst = np.float32([                              # Destination square
        [0,0],
        [res[0]-1,0],
        [res[0]-1,res[1]-1],
        [0,res[1]-1]
    ])

    h, _ = cv2.findHomography(src, dst)             # Find transformation matrix
    
    new_p = np.dot((point[0], point[1], 1), h.T)    # Get new point

    new_p = new_p/new_p[-1]                         # Remove omega value

    return new_p[:-1]

