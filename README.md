# Inference with [YOLOv5](https://github.com/ultralytics/yolov5), ArUco markers, and perspective correction.

Tested with python 3.8.10.
Installed torch with:
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## To use:
Generate markers with create_markers.py.
Place markers in a square following this pattern:
<pre>
id0-----id1<br />
|         |<br /> 
|         |<br /> 
id4-----id3
</pre>
Update path in yolo_inference.py, line 25, with path to custom weights.
Run yolo_inference.py to start.

### Init:
  1) Loads trained weights from a [YOLOv5](https://github.com/ultralytics/yolov5) network.
### Mainloop
  1) Reads a camera frame with opencv.
  2) Object detection inference.
  3) Draws rectangles on frame.
  4) Transforms detection from image to map by ArUco markers.
  5) Displays frame and new coordinate system.
