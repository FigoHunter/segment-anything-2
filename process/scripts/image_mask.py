import argparse
import os
import glob
import cv2
import numpy as np
from threading import Lock
from interactive_sam.history import History
from interactive_sam import mask_gen

MAX_HISTORY_SIZE = 100
ALPHA = 0.5

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

selection = [[],[]]
img_file = None
mask = None
mask_colored = None

interface_lock = Lock()

def mouse_callback(event, x, y, flags, param):
    global img_file, mask, mask_colored

    if event == cv2.EVENT_LBUTTONUP:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            interface_lock.acquire()
            selection[1].append([x, y])
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            interface_lock.acquire()
            selection[0].append([x, y])
        else:
            return
        history.register(selection)
        mask = mask_gen.load_image(img_file, selection[0], selection[1])
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored = (mask_colored * 255).astype(np.uint8)
        interface_lock.release()

wnd = cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('image', mouse_callback)

history = History(selection, MAX_HISTORY_SIZE)


for image in images:
    print(f"Processing {image}")
    mask_path = image.replace('rgb', 'masks')
    org_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img_file=image
    while True:
        img = org_img.copy()
        if mask_colored is not None:
            img = cv2.addWeighted(mask_colored, ALPHA, img, 1 - ALPHA, 0)


        for (x, y) in selection[1]:
            cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        for (x, y) in selection[0]:
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        
        # 按下 左
        if key == 81:
            result = history.undo(False)
            if result is None:
                print("Nothing to undo")
            else:
                interface_lock.acquire()
                selection = result
                mask = mask_gen.load_image(img_file, selection[0], selection[1])
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_colored = (mask_colored * 255).astype(np.uint8)
                interface_lock.release()
        
        # 按下 右
        elif key == 83:
            result = history.redo(False)
            if result is None:
                print("Nothing to redo")
            else:
                interface_lock.acquire()
                selection = result
                mask = mask_gen.load_image(img_file, selection[0], selection[1])
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_colored = (mask_colored * 255).astype(np.uint8)
                interface_lock.release()
        
        # 按下 空格
        if key == 27:
            break

        # 按下 Esc
        if key == 32:
            exit(0)

        # 按下 Enter
        if key == 13:
            print(f'Saving mask to {mask_path}')
            if os.path.exists(mask_path):
                print(f"Mask {mask_path} already exists. Skipping")
                continue
            interface_lock.acquire()
            mask_bw = (mask * 255).astype(np.uint8)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, mask_bw)
            interface_lock.release()
            break

    
    cv2.destroyAllWindows()
