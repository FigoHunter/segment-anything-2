import argparse
import cv2
import os
from enum import Enum
from threading import Lock, Thread
from interactive_sam.mask_gen import init_sam2_video, Sam2VideoHandle
from interactive_sam import keyboard,FrameWait
from interactive_sam.operation import Operation
from interactive_sam.color import get_color_from_set
import numpy as np

MAX_HISTORY_SIZE = 100
ALPHA = 0.3
SEL_APLHA = 0.6

parser  = argparse.ArgumentParser(description='Test sam2 videos')
parser.add_argument('path', type=str, help='Path to the videos')
args = parser.parse_args()
path = args.path

class CacheImageReader:
    def __init__(self, cache_size=15):
        self.cache = {}
        self._load_lock = Lock()
        self._cache_size = cache_size
        # self._queue_count=0

    def _cache(self, path, img):
        if len(self.cache) >= self._cache_size:
            self.cache.popitem()
        self.cache[path] = img

    def read(self, path):
        self._load_lock.acquire()
        # self._queue_count += 1

        if path in self.cache:
            img = self.cache[path]
            # self._queue_count -= 1
            self._load_lock.release()
            return img
        img = cv2.imread(path)
        self.cache[path] = img
        # self._queue_count -= 1
        self._load_lock.release()
        return img

    def _async_read(self,  paths, idx, start_idx, end_idx):
        prev_count = idx - start_idx
        next_count = end_idx - idx
        for i in range(max(prev_count, next_count)):
            if i < next_count:
                self.read(paths[idx + i + 1])
            if i < prev_count:
                self.read(paths[idx - i - 1])

    def read_with_preload(self, paths, idx=0, *, r=2, thread=True):
        start_idx = max(0, idx-r)
        end_idx = min(len(paths)-1, idx+r)
        if not thread:
            for i in range(start_idx, end_idx + 1):
                self.read(paths[idx+i])
            return self.read(paths[idx])
        else:
            img = self.read(paths[idx])
            Thread(target=self._async_read, args=(paths, idx, start_idx, end_idx), daemon=True).start()
            return img 


def render_points(img, include_points, exclude_points):
    if include_points is not None:
        for p in include_points:
            cv2.circle(img, tuple(p), 7, (0, 255, 0), -1)
    if exclude_points is not None:
        for p in exclude_points:
            cv2.circle(img, tuple(p), 7, (0, 0, 255), -1)

    return img

def rgb2bgr(color):
    return color[2], color[1], color[0]

def render_frame(base_img, task:Sam2VideoHandle.Task, *, sel_alpha=SEL_APLHA, alpha=ALPHA):
    img = base_img.copy()
    objs, masks = task.get_current_masks()
    print('objs:', objs)
    print('masks:', [mask.shape for mask in masks])
    for obj, mask in zip(objs, masks):
        if mask is None:
            continue
        selected = task.get_current_obj()
        if selected == obj:
            a = sel_alpha
        else:
            a = alpha
        color = get_color_from_set(obj)
        mask_colored = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255 * rgb2bgr(color)).astype(np.uint8)
        img = cv2.addWeighted(img, 1 - alpha, mask_colored, a, 0)
    selection = task.get_selection()
    if selection is not None:
        include_points, exclude_points = selection
    else:
        include_points, exclude_points = None, None
    render_points(img, include_points, exclude_points)
    return img
    

def main():
    image_reader = CacheImageReader()
    obj_list = []
    while True:
        obj_idx = input('Enter object index: ')
        if obj_idx == '':
            break
        if obj_idx.isdigit():
            obj_list.append(int(obj_idx))
        else:
            print('int supported only')

    key_handler = keyboard.get_key_handler()
    key_handler.start()

    handle = init_sam2_video()
    print('init:', handle)
    rgb_path = os.path.join(path, 'rgb')
    tasks = handle.load_frames(rgb_path,obj_list, chunk_size=300)
    frame_wait = FrameWait(0.5)

    for task in tasks:
        if len(task.frames)<1:
            print('No frames found in task. Skipping')
            print('===============================')
            continue
        # print('frames:', task.frames)
        task.start()
        rendered_img = None
        interface_lock = Lock()
        org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
        propagate = None
        paused = False
        reverse = None
        rendered_img = render_frame(org_img,task)
        flag_labeling = False
        is_dirty = False
        next_task = False
        key_handler.clear()


        @key_handler.register_op_wrap(Operation.NEXT_IMG)
        def next_image(key):
            interface_lock.acquire()
            nonlocal propagate, paused
            if propagate is not None:
                paused = True
            if flag_labeling:
                print('Please press ENTER finish label')
                return
            nonlocal rendered_img
            task.next_frame()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()


        @key_handler.register_op_wrap(Operation.PREV_IMG)
        def prev_image(key):
            interface_lock.acquire()
            nonlocal propagate, paused
            if propagate is not None:
                paused = True
            nonlocal rendered_img
            task.prev_frame()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.NEXT_OBJ)
        def next_obj(key):
            interface_lock.acquire()
            nonlocal propagate, paused
            if propagate is not None:
                paused = True
            nonlocal rendered_img
            task.next_obj()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            print('Selected object:', task.get_current_obj())
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.PREV_OBJ)
        def prev_obj(key):
            interface_lock.acquire()
            nonlocal propagate, paused
            if propagate is not None:
                paused = True
            nonlocal rendered_img
            task.prev_obj()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            print('Selected object:', task.get_current_obj())
            interface_lock.release()

        @key_handler.register_key_wrap(keyboard.Keys.ENTER, keyboard.Event.PRESS)
        def on_enter(key):        
            nonlocal flag_labeling, propagate, is_dirty, rendered_img
            if flag_labeling:
                try:
                    interface_lock.acquire()
                    propagate = None    
                    if is_dirty:
                        task.update_selection()
                        is_dirty = False
                        org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
                        rendered_img = render_frame(org_img,task)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(e)
                    interface_lock.release()
                    return
                print('Labeling finished')
                flag_labeling = False
                interface_lock.release()
            else:
                flag_labeling = True
                print('Labeling started')

        def mouse_callback(event, x, y, flags, param):
            nonlocal rendered_img, flag_labeling, is_dirty, propagate
            try:
                if event == cv2.EVENT_LBUTTONUP:
                    if flags & cv2.EVENT_FLAG_CTRLKEY:
                        interface_lock.acquire()
                        propagate = None                        
                        flag_labeling = True
                        print('Labeling started')
                        selection = task.get_selection()
                        if selection is None:
                            selection = [[],[]]
                        selection[1].append([x, y])
                        task.set_selection(selection)
                        is_dirty = True
                        task.register_history()
                    elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                        interface_lock.acquire()
                        propagate = None
                        flag_labeling = True
                        print('Labeling started')
                        selection = task.get_selection()
                        if selection is None:
                            selection = [[],[]]
                        selection[0].append([x, y])
                        task.set_selection(selection)
                        is_dirty = True
                        task.register_history()
                    else:
                        return
                    org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
                    rendered_img = render_frame(org_img,task)
                    interface_lock.release()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                exit(1)

        @key_handler.register_op_wrap(Operation.CLEAR)
        def clear(key):
            nonlocal propagate
            interface_lock.acquire()
            if not flag_labeling:
                print('Please press ENTER start label')
                return
            propagate = None
            print('Clearing selection')
            task.clear_frame()
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.QUIT)
        def quit(key):
            exit(0)

        @key_handler.register_key_wrap(keyboard.Keys.LEFT | keyboard.Keys.CTRL_L, keyboard.Event.PRESS)
        def fast_left(key):
            nonlocal propagate, rendered_img, paused
            interface_lock.acquire()
            if propagate is not None:
                paused = True
            if flag_labeling:
                print('Please press ENTER finish label')
                return
            if not task.fast_left():
                print('No more frames')
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()

        @key_handler.register_key_wrap(keyboard.Keys.RIGHT | keyboard.Keys.CTRL_L, keyboard.Event.PRESS)
        def fast_right(key):
            nonlocal propagate, rendered_img, paused
            interface_lock.acquire()
            if propagate is not None:
                paused = True
            if flag_labeling:
                print('Please press ENTER finish label')
                return
            if not task.fast_right():
                print('No more frames')
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.UNDO)
        def undo(key):
            nonlocal rendered_img, propagate, flag_labeling, is_dirty
            interface_lock.acquire()
            propagate = None
            task.undo()
            flag_labeling = True
            is_dirty = True
            print('Start labeling')
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.REDO)
        def redo(key):
            nonlocal rendered_img,propagate, flag_labeling
            interface_lock.acquire()
            propagate = None
            task.redo()
            flag_labeling = True
            is_dirty = True
            print('Start labeling')
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            rendered_img = render_frame(org_img,task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.SAVE)
        def save(key):
            nonlocal propagate
            if flag_labeling:
                print('Please press ENTER finish label')
                return
            interface_lock.acquire()
            propagate = None
            task.save_files()
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.PROCESS)
        def process(key):
            nonlocal propagate, flag_labeling, paused, reverse
            interface_lock.acquire()
            if flag_labeling:
                print('Please press ENTER finish label')
                interface_lock.release()
                return
            if propagate is not None:
                if not paused:
                    if reverse:
                        print('Cannot revert while propagating')
                        interface_lock.release()
                        return
                    paused = True
                    print('Paused propagation')
                    interface_lock.release()
                    return
                else:
                    if not reverse:
                        paused = False
                        print('Resuming propagation')
                        interface_lock.release()
                        return
            if not task.is_keyframe():
                print('Not keyframe')
                interface_lock.release()
                return
            paused = False
            propagate = task.propagate()
            reverse = False 
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.PROCESS_REVERT)
        def process_revert(key):
            nonlocal propagate, flag_labeling, paused, reverse
            interface_lock.acquire()
            if flag_labeling:
                print('Please press ENTER finish label')
                interface_lock.release()
                return
            if propagate is not None:
                if not paused:
                    if not reverse:
                        print('Cannot revert while propagating')
                        interface_lock.release()
                        return
                    paused = True
                    print('Paused propagation')
                    interface_lock.release()
                    return
                else:
                    if reverse:
                        paused = False
                        print('Resuming propagation')
                        interface_lock.release()
                        return
            if not task.is_keyframe():
                print('Not keyframe')
                interface_lock.release()
                return
            paused = False
            propagate = task.propagate(reverse=True)
            reverse = True
            interface_lock.release()

        @key_handler.register_key_wrap(keyboard.Keys.SPACE, keyboard.Event.PRESS)
        def stop_propagation(key):
            nonlocal propagate, paused
            interface_lock.acquire()
            if propagate is not None:
                print('Paused propagation')
                paused = True
            interface_lock.release()

        @key_handler.register_key_wrap(keyboard.Keys.CTRL_R | keyboard.Keys.PAGE_DOWN, keyboard.Event.PRESS)
        def press_next_task(key):
            nonlocal next_task
            while True:
                i = input('Skipping To Next Task.(y/n): ')
                if i == 'y' or i == 'Y':
                    next_task = True
                    break
                elif i == 'n' or i == 'N':
                    next_task = False
                    break

        wnd = cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('image', mouse_callback)
        while True:
            interface_lock.acquire()
            if propagate is not None:
                if flag_labeling:
                    propagate = None
                    interface_lock.release()
                    continue
                if not paused:
                    print('Propagating')
                    # frame_wait.start_frame()
                    try:
                        next(propagate)
                    except StopIteration:
                        propagate = None
                        # frame_wait.wait_frame()
                        # interface_lock.release()
                # frame_wait.wait_frame()
                org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
                rendered_img = render_frame(org_img,task)
            interface_lock.release()
            cv2.imshow('image', rendered_img)
            objs, masks = task.get_current_masks()
            # if len(masks) > 0:
            #     mask_colored = (cv2.cvtColor(masks[0], cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
            #     cv2.imshow('mask', mask_colored)
            # if temp is not None:
            #     cv2.imshow('temp', temp)
            # time.sleep(0.1)
            cv2.waitKey(100)
            if next_task:
                break

            
if __name__ == '__main__':
    main()