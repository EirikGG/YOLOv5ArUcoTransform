import torch, cv2, datetime

import numpy as np

from PIL import Image
from transform_coordinates import get_coords

def add_border(img, size=(10, 10, 10, 10)):
    '''Adds borders to an image '''
    return cv2.copyMakeBorder(
        img,
        *size,
        borderType=cv2.BORDER_CONSTANT
    )

def get_empty_img(map_size = (1000, 1000)):
    '''Returns an empty white image'''
    map_img = np.zeros((map_size[0], map_size[1],3), np.uint8)
    map_img.fill(255)
    return map_img


yolov5s = torch.hub.load(                               # Load yolo model
    'ultralytics/yolov5', 
    'custom', 
    'weights/best.pt'
)

#yolov5s = yolov5s.fuse().autoshape()

cap = cv2.VideoCapture(0)

map_size = (1000, 1000)                                 # Initialize map
map_img = get_empty_img(map_size)

while True:
    _, frame = cap.read()                               # Input frame
    #frame = cv2.imread('testImg/markers.png')

    out = yolov5s(frame)                                # Get predictions

    out_df = out.pandas().xyxy[0]                       # Read predictions to pandas

    out_df = out_df[.9<out_df['confidence']]            # Filter confidence

    if not out_df.empty:                                # Draw rectangles to image
        for i, row in out_df.iterrows():
            frame = cv2.rectangle(
                frame,
                (int(row['xmin']), int(row['ymin'])),
                (int(row['xmax']), int(row['ymax'])),
                (0, 255, 0),
                1
            )

        out_df = out_df.reset_index()                   # Set numerical index
                                                        # Get best case
        max_thres = out_df.loc[out_df['confidence'].idxmax()]

        center_point = (
            int(max_thres['xmin'] + (max_thres['xmax'] - max_thres['xmin'])/2),
            int(max_thres['ymin'] + (max_thres['ymax'] - max_thres['ymin'])/2)
        )

                                                        # Translate from image to map
        new_coords, aruco_points, h = get_coords(frame, center_point, new_coord_res=map_size)

        if any(new_coords):                             # If successfull tranlation, draw map
            map_img = get_empty_img(map_size)
            map_img = cv2.circle(map_img, new_coords, radius=0, color=(0, 0, 255), thickness=20)

            for i in aruco_points:
                frame = cv2.putText(frame, str(i), aruco_points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

                                                        # Rezise map to fit frame
    map_img = cv2.resize(map_img, (frame.shape[0], frame.shape[0])) 
    numpy_horizontal = np.hstack((
        add_border(frame),
        add_border(map_img)
    ))

    cv2.imshow('preview', numpy_horizontal)             # Show both images

    key = cv2.waitKey(1)                                # Wait for user to type 

    if 115 == key:                                      # Press s to save image
        rgb_img = cv2.cvtColor(numpy_horizontal, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        now = datetime.datetime.now()                   # Create timestamped name
        current_timestamp = now.strftime("%Y%m%d-%H%M%S")
        image_name = f'image{current_timestamp}.jpg'

        pil_img.save(image_name)
        print(f'Saved image as {image_name}')


    elif 113 == key: break                              # "q" to break

    
cap.release()
cv2.destroyAllWindows()