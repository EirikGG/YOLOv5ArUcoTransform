import cv2
import statistics as stat
import numpy as np


def get_aruco_pos(img, dictionary=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)):
    '''Detects aruco markers in cv2 input image and returns dictionary with id and 
    center points. If no markers are found returns None'''

    # Detects markers based on image and dictionary with standars parameters
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        dictionary,
        parameters=cv2.aruco.DetectorParameters_create()
    )

    ps = {}
    for i, sq in enumerate(markerCorners):
        sq = sq[0]                                  # Unpacks square
        ID = str(*markerIds[i])                     # Gets markers id

        cen_x = int(stat.mean([x[0] for x in sq]))  # Centerpoint is mean value
        cen_y = int(stat.mean([y[1] for y in sq]))

        ps[int(ID)] = (cen_x, cen_y)                # Save to dict id: (x, y)


    return ps if 4==len(ps) else None               # If square is not found, return None


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

    return *new_p[:-1], h

def get_coords(img, point2d, new_coord_res=(100,100)):
    aruco_points = get_aruco_pos(img)

    x, y, h = None, None, None
    if aruco_points:
        x, y, h = transform_to_new_coord(
            point=point2d,
            pixel_coords=(
                aruco_points[0],
                aruco_points[1],
                aruco_points[2],
                aruco_points[3]),
            res=new_coord_res
        )

    if any((x, y)):
        x,y = tuple(map(int, (x, y)))
    
    return (x, y), aruco_points, h



if '__main__'==__name__:
    img = cv2.imread('markers.png')

    print(get_coords(img, (430, 451), (1000, 1000)))