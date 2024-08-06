import argparse
import os
import glob
import cv2
from interactive_sam.utils import History

MAX_HISTORY_SIZE = 100

parser = argparse.ArgumentParser(description='Extract masks from images')
parser.add_argument('path', type=str, help='Path to the image')
# parser.add_argument('--mode','-m', choices=['label', 'process'], default=None, help='Mode of operation')
args = parser.parse_args()

# mode = args.mode
path = args.path

assert os.path.exists(path), f"Path {path} does not exist"
if os.path.isdir(path):
    data_path = os.path.abspath(path)
    images = glob.glob(os.path.join(data_path, '*/rgb/000001.png'))
else:
    data_path = os.path.dirname(path)
    images = [path]

# if mode is None:
#     print("Please specify the mode of operation [label, process]")
#     exit(1)

assert len(images) > 0, f"No images found in {data_path}"
print(images)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Left button up at ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONUP:
        print(f"Right button up at ({x}, {y})")


wnd = cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', mouse_callback)

selection = [[],[]]
history = History(selection, MAX_HISTORY_SIZE)


for image in images:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    
    # 按下 左
    if key == 81:
        result = history.undo(selection,False)
        if result is None:
            print("Nothing to undo")
        else:
            selection = result
            print("Undo")
    
    # 按下 右
    elif key == 83:
        result = history.redo(selection,False)
        if result is None:
            print("Nothing to redo")
        else:
            selection = result
            print("Redo")
