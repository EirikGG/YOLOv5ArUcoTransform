import cv2

from detect_aruco import get_aruco_pos
from homography import transform_to_new_coord

def get_coords(img, point2d):
    aruco_points = get_aruco_pos(img)

    x, y = None, None
    if aruco_points:
        x, y = transform_to_new_coord(
            point=point2d,
            pixel_coords=(
                aruco_points[0],
                aruco_points[1],
                aruco_points[2],
                aruco_points[3]),
            res=(100,100)
        )
    
    return x,y



if '__main__'==__name__:
    img = cv2.imread('IMG_20210417_203220.jpg')

    print(get_coords(img, (430, 451)))