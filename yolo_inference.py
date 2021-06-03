import torch, cv2, datetime, imutils, os

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

def get_timestamp_str():
    '''Gets timestamp as a string'''
    now = datetime.datetime.now()                   # Create timestamped name
    return now.strftime("%Y%m%d-%H%M%S")


def save_img_timestamp(img, name):
    '''Saves an image with timestamped name'''
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pil_img.save(name)
    print(f'Saved image as {name}')

def save_str_timestamp(text_name, **kwargs):
    '''Takes a textfile and saves input as key and value'''
    with open(text_name, 'w') as f:
        for key, value in kwargs.items():
            f.writelines(f'{key}: {value}\n')


yolov5x = torch.hub.load(                               # Load yolo model
    'ultralytics/yolov5', 
    'custom', 
    'weights/best.pt'
)

#yolov5s = yolov5s.fuse().autoshape()

cap = cv2.VideoCapture(0)

map_size = (1000, 1400)                                 # Initialize map (height, width)
map_img = get_empty_img(map_size)

while True:
    #_, frame = cap.read()                               # Input frame
    frame = cv2.imread('testImgs/IMG_20210603_153639.jpg')

    out = yolov5x(frame)                                # Get predictions

    out_df = out.pandas().xyxy[0]                       # Read predictions to pandas

    out_df = out_df[.5<out_df['confidence']]            # Filter confidence

    obj_coords = None
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

        frame = cv2.circle(frame, center_point, radius=0, color=(0, 0, 255), thickness=20)
                                                        # Translate from image to map
        new_coords, aruco_points, h = get_coords(frame, center_point, new_coord_res=map_size)
        obj_coords = new_coords

        if any(new_coords):                             # If successfull tranlation, draw map
            map_img = get_empty_img(map_size)
            map_img = cv2.circle(map_img, new_coords, radius=0, color=(0, 0, 255), thickness=20)
            for i in aruco_points:
                frame = cv2.putText(frame, str(i), aruco_points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

                                                        # Rezise map to fit frame
    map_img = cv2.resize(map_img, (frame.shape[0], frame.shape[0]))
    #map_img = imutils.resize(map_img, height=frame.shape[0])

    numpy_horizontal = np.hstack((
        add_border(frame),
        add_border(map_img)
    ))

    numpy_horizontal = imutils.resize(numpy_horizontal, height=1000)

    cv2.imshow('preview', numpy_horizontal)             # Show both images

    key = cv2.waitKey(1)                                # Wait for user to type 

    if 115 == key:                                      # Press "s" to save image
        dirname = 'save'
        if not os.path.isdir(dirname): os.mkdir(dirname)
        time_stamp = get_timestamp_str()
        save_img_timestamp(
            numpy_horizontal,
            os.path.join(dirname, f'image{time_stamp}.jpg')
        )
        save_str_timestamp(
            os.path.join(dirname, f'coords{time_stamp}.txt'),
            map_size = map_size,
            coordinates = obj_coords
        )


    elif 113 == key: break                              # "q" to break

    
cap.release()
cv2.destroyAllWindows()