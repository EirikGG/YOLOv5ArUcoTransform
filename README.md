# Inference with [YOLOv5](https://github.com/ultralytics/yolov5), ArUco markers, and perspective correction.

Tested with python 3.8.10.

## To use:
Update path in line 25 in yolo_inference.py with path to custom weights.
Run yolo_inference.py to start.

### Init:
  1) Loads trained weights from a [YOLOv5](https://github.com/ultralytics/yolov5) network.
### Mainloop
  1) Reads a camera frame with opencv.
  2) Object detection inference.
  3) Draws rectangles on frame.
  4) Transforms detection from image to map by ArUco markers.
  5) Displays frame and new coordinate system.
