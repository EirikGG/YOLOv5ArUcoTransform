import cv2
import statistics as stat


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