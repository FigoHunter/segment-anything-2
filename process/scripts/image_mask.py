import argparse
import os
import glob
import cv2
import numpy as np
from threading import Lock
from interactive_sam.history import History
from interactive_sam import mask_gen,keyboard
from interactive_sam.operation import Operation
from interactive_sam.utils import save_pkl,load_pkl

MAX_HISTORY_SIZE = 100
ALPHA = 0.5
ITERATIONS = 1



def main():
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
    mask = None
    org_img = None
    rendered_img = None
    history = History(selection, MAX_HISTORY_SIZE)

    interface_lock = Lock()
    handle = mask_gen.init_sam2_image()
    handler = keyboard.get_key_handler()
    image_id = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal mask
        img_file = images[image_id]

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
            mask = mask_gen.load_image(handle, img_file, selection[0], selection[1], ITERATIONS)
            render_image()
            interface_lock.release()

    @handler.register_op_wrap(Operation.CLEAR)
    def clear(key):
        print('Clearing selection')
        nonlocal selection, mask
        interface_lock.acquire()
        selection = [[], []]
        history.register(selection)
        mask = None
        render_image()
        interface_lock.release()

    @handler.register_op_wrap(Operation.QUIT)
    def quit(key):
        cv2.destroyAllWindows()
        handler.stop()
        exit(0)

    @handler.register_op_wrap(Operation.NEXT_IMG)
    def next_image(key):
        nonlocal image_id
        if image_id < len(images) - 1:
            image_id = (image_id + 1) % len(images)
            reset_image()
            render_image()
        
    @handler.register_op_wrap(Operation.PREV_IMG)
    def prev_image(key):
        nonlocal image_id
        if image_id > 0:
            image_id = (image_id - 1) % len(images)
            reset_image()
            render_image()

    @handler.register_op_wrap(Operation.UNDO)
    def undo(key):
        nonlocal selection, mask
        img_file = images[image_id]
        result = history.undo(False)
        if result is None:
            print("Nothing to undo")
        else:
            interface_lock.acquire()
            selection = result
            mask = mask_gen.load_image(handle, img_file, selection[0], selection[1], ITERATIONS)
            render_image()
            interface_lock.release()

    @handler.register_op_wrap(Operation.REDO)
    def redo(key):
        nonlocal selection, mask
        img_file = images[image_id]
        result = history.redo(False)
        if result is None:
            print("Nothing to redo")
        else:
            interface_lock.acquire()
            selection = result
            mask = mask_gen.load_image(handle, img_file, selection[0], selection[1], ITERATIONS)
            render_image()
            interface_lock.release()

    @handler.register_op_wrap(Operation.SAVE)
    def save(key):
        print('Saving mask')
        nonlocal mask, selection
        mask_path = images[image_id].replace('rgb', 'masks')
        print(f'Saving mask to {mask_path}')
        interface_lock.acquire()
        mask_bw = (mask * 255).astype(np.uint8)
        print(mask_bw.dtype)  # 检查初始图像的类型
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask_bw)
        save_pkl(selection, mask_path, 'selection')
        interface_lock.release()

    def reset_image():
        nonlocal selection, mask, history
        mask_path = images[image_id].replace('rgb', 'masks')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            print(mask.max())
            selection = load_pkl(mask_path, 'selection', [[], []])
        history = History(selection, MAX_HISTORY_SIZE)

    def render_image():
        nonlocal org_img, rendered_img, mask
        image = images[image_id]
        org_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        rendered_img = org_img.copy()
        if mask is not None:
            mask_colored = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
            rendered_img = cv2.addWeighted(mask_colored, ALPHA, rendered_img, 1 - ALPHA, 0)

        for (x, y) in selection[1]:
            cv2.circle(rendered_img, (x, y), 10, (0, 0, 255), -1)
        for (x, y) in selection[0]:
            cv2.circle(rendered_img, (x, y), 10, (0, 255, 0), -1)


    wnd = cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('image', mouse_callback)

    handler.start()
    reset_image()
    render_image()
    while True:
        cv2.imshow('image', rendered_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()