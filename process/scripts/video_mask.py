import argparse
import cv2
from threading import Lock, Thread, RLock, Event as TEvent
from interactive_sam.mask_gen import init_sam2_video, Sam2VideoHandle
from interactive_sam import keyboard
from interactive_sam.operation import Operation
from interactive_sam.history import History
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
    LOAD_SIZE = 3
    def __init__(self, cache_size=10):
        self.cache = {}
        self._load_lock = Lock()
        self._cache_size = cache_size
        self._queue_count=0

    def _cache(self, path, img):
        if len(self.cache) >= self._cache_size:
            self.cache.popitem()
        self.cache[path] = img


    def read(self, path):
        self._load_lock.acquire()
        self._queue_count += 1

        if path in self.cache:
            img = self.cache[path]
            self._queue_count -= 1
            self._load_lock.release()
            return img
        img = cv2.imread(path)
        self.cache[path] = img
        self._queue_count -= 1
        self._load_lock.release()
        return img

    def _async_read(self,  paths, idx, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            if i == idx:
                continue
            if self._queue_count > self.LOAD_SIZE:
                continue
            self.read(paths[i])



    def read_with_preload(self, paths, idx=0, *, range=2, thread=True):
        start_idx = max(0, idx-range)
        end_idx = min(len(paths), idx+range+1)
        if not thread:
            for i in range(start_idx, end_idx):
                self.read(paths[idx+i])
            return self.read(paths[idx])
        else:
            img = self.read(paths[idx])
            Thread(target=self._async_read, args=(paths, idx, start_idx, end_idx)).start()
            return img 


def render_points(img, include_points, exclude_points):
    if include_points is not None:
        for p in include_points:
            cv2.circle(img, tuple(p), 2, (0, 255, 0), -1)
    if exclude_points is not None:
        for p in exclude_points:
            cv2.circle(img, tuple(p), 2, (0, 0, 255), -1)

    return img

def rgb2bgr(color):
    return color[2], color[1], color[0]

def render_frame(base_img, task:Sam2VideoHandle.Task, *, sel_alpha=SEL_APLHA, alpha=ALPHA):
    img = base_img.copy()
    objs, masks = task.get_current_masks()
    for obj, mask in zip(objs, masks):
        if mask is None:
            continue
        if task.prompts.selected_obj == obj:
            a = sel_alpha
        else:
            a = alpha
        color = get_color_from_set(obj)
        mask_colored = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * rgb2bgr(color)).astype(np.uint8)
        img = cv2.addWeighted(img, 1 - a, mask_colored, a, 0)
    render_points(img, task.prompts.get_selection())
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
    handle = init_sam2_video()
    print('init:', handle)
    tasks = handle.load_frames(path,obj_list)
    
    for task in tasks:
        if len(task.frames)<1:
            print('No frames found in task. Skipping')
            print('===============================')
            continue
        task.start()
        rendered_img = None
        interface_lock = Lock()
        history = History(task.prompts, MAX_HISTORY_SIZE)
        org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
        propagate = None
        render_frame(org_img,task)

        @key_handler.register_op_wrap(Operation.NEXT_IMG)
        def next_image(key):
            interface_lock.acquire()
            task.next_frame()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            render_frame(org_img, task)
            interface_lock.release()


        @key_handler.register_op_wrap(Operation.PREV_IMG)
        def prev_image(key):
            interface_lock.acquire()
            task.prev_frame()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            render_frame(org_img, task)
            interface_lock.release()


        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    next_image(None)
                else:
                    prev_image(None)

            if event == cv2.EVENT_LBUTTONUP:
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    interface_lock.acquire()
                    frame,obj,selection = task.prompts.get_selection()
                    if selection is None:
                        selection = [[],[]]
                    selection[1].append([x, y])
                    task.prompts.set_selection(selection)
                    task.update_selection()
                elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                    interface_lock.acquire()
                    frame,obj,selection = task.prompts.get_selection()
                    if selection is None:
                        selection = [[],[]]
                    selection[0].append([x, y])
                    task.prompts.set_selection(selection)
                    task.update_selection()
                else:
                    return
                history.register(task.prompts)
                org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
                render_frame(org_img, task)
                interface_lock.release()
                

        @key_handler.register_op_wrap(Operation.CLEAR)
        def clear(key):
            print('Clearing selection')
            interface_lock.acquire()
            frame, obj, selection = task.prompts.get_selection()
            if selection is None:
                interface_lock.release()
                return
            selection = None
            task.prompts.set_selection(selection)
            task.update_selection()
            history.register(task.prompts)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.QUIT)
        def quit(key):
            cv2.destroyAllWindows()
            key_handler.stop()
            exit(0)

        @key_handler.register_op_wrap(Operation.UNDO)
        def undo(key):
            interface_lock.acquire()
            task.prompts = history.undo()
            task.update_selection()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            render_frame(org_img, task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.REDO)
        def redo(key):
            interface_lock.acquire()
            task.prompts = history.redo()
            task.update_selection()
            org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
            render_frame(org_img, task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.SAVE)
        def save(key):
            interface_lock.acquire()
            task.save()
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.PROCESS)
        def process(key):
            nonlocal propagate
            interface_lock.acquire()
            propagate = task.propagate()
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.NEXT_OBJ)
        def next_obj(key):
            interface_lock.acquire()
            task.next_obj()
            render_frame(org_img, task)
            interface_lock.release()

        @key_handler.register_op_wrap(Operation.PREV_OBJ)
        def prev_obj(key):
            interface_lock.acquire()
            task.prev_obj()
            render_frame(org_img, task)
            interface_lock.release()

        wnd = cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('image', mouse_callback)
        while True:
            if propagate is not None:
                print('Propagating')
                try:
                    next(propagate)
                except StopIteration:
                    propagate = None
                    continue
                org_img = image_reader.read_with_preload(task.frames, task.get_current_frame_idx())
                render_frame(org_img, task)

            cv2.imshow('image', rendered_img)
            cv2.waitKey(30)

            
if __name__ == '__main__':
    main()