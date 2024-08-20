import os
import cv2
import shutil
import numpy as np
from enum import Enum
from queue import Queue as Q
from ycb_objects import load
from threading import Thread, RLock, Event


class AsyncCacheSave:
    CACHE_COUNT = 20
    THREAD_COUNT = 4
    
    def __init__(self, paths):
        self._workers = [None] * 4
        for i in range(len(self._workers)):
            self._workers[i] = Thread(target=self._run, args=(i,), daemon=True)
        self._q = Q()
        self._q_lock = RLock()
        self._q_dict = {}
        self._cached_data = {}
        self._abort = set()
        self._paths = [os.path.splitext(path.replace('rgb','video_masks'))[0] for path in paths]
        self._temp_dirs = [self._get_temp_path(i) for i in range(len(paths))]
        for tmp_path in self._temp_dirs:
            shutil.rmtree(os.path.dirname(tmp_path), ignore_errors=True)


    class CacheHandle:
        def __init__(self):
            self._event = Event()
        def wait(self):
            self._event.wait()
        def _set(self):
            self._event.set()
    
    class CacheType:
        PATH=0
        WEAK=1
        STRONG=2

    def start(self):
        for worker in self._workers:
            worker.start()

    def stop(self):
        handle = self.CacheHandle()
        self._q.empty()
        self._q.put(('quit', None, handle))
        return handle

    def _get_temp_path(self, frame_id):
        dir = os.path.dirname(self._paths[frame_id])
        basename = os.path.basename(self._paths[frame_id])
        return os.path.join(dir, 'temp', basename)

    def update_cache(self, objs, masks, frame_id):
        data = (objs, masks)
        handle = self.CacheHandle()
        self._q_lock.acquire()
        self._abort_q(frame_id)
        self._add_q_dict(handle, frame_id)
        self._add_cache(data, frame_id)
        self._q_lock.release()
        self._q.put(('save', frame_id, handle))
        return handle

    def delete_cache(self, frame_id):
        handle = self.CacheHandle()
        self._abort_q(frame_id)
        self._delete(frame_id)
        # self._q.put(('delete', frame_id, handle))
        return handle        

    def save_files(self):
        self._q_lock.acquire()
        self._q.join()
        for frame_id in self._cached_data:
            if self._cached_data[frame_id][1] > self.CacheType.WEAK:
                raise Exception(f"Frame {frame_id} is not saved to temp Path")
            if self._cached_data[frame_id][1] == self.CacheType.PATH:
                path = self._cached_data[frame_id][0]
            elif self._cached_data[frame_id][1] == self.CacheType.WEAK:
                path = self._temp_dirs[frame_id]
            for obj in os.listdir(path):
                obj_path = os.path.join(path, obj)
                if os.path.isdir(obj_path):
                    continue
                obj = int(os.path.splitext(obj)[0])
                dirname = os.path.dirname(self._paths[frame_id])
                basename = os.path.basename(self._paths[frame_id])
                frame = int(basename)
                obj = load.get_object_name(obj)
                save_path = os.path.join(dirname, obj, f'{frame:06d}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print('saving by copy: ', obj_path, save_path)
                shutil.copy(obj_path, save_path)
        self._q_lock.release()
            

    def __del__(self):
        self.stop()
        self._worker.join()

    def _run(self, thread_id):
        while True:
            process = self._q.get()
            if process[0] == 'quit':
                break
            handle = process[2]
            if process[0] == 'save':
                self._q_lock.acquire()
                if handle in self._abort:
                    self._abort.pop(handle)
                    self._q_dict[process[1]].remove(handle)
                    self._q_lock.release()
                    continue
                try:
                    self._save(process[1], handle)
                    self._overfill_cache(process[1])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(e)
                    print(f'[THREAD-{thread_id}] Error saving {process[1]}')
                self._q_lock.release()
                self._q.task_done()
            # elif process[0] == 'delete':
            #     self._del(process[1])
            handle._set()

    def _save(self, frame_id, handle):
        self._q_lock.acquire()
        data = self._cached_data[frame_id][0]
        path = self._temp_dirs[frame_id]
        shutil.rmtree(path, ignore_errors=True)
        objs = data[0]
        masks = data[1]
        for obj, mask in zip(objs, masks):
            sub_path = os.path.join(path, f'{obj}.png')
            os.makedirs(os.path.dirname(sub_path), exist_ok=True)
            cv2.imwrite(sub_path, (mask * 255).astype(np.uint8))
        self._q_dict[frame_id].remove(handle)
        self._cached_data[frame_id][1] = self.CacheType.WEAK
        self._q_lock.release()

    def _delete(self, frame_id):
        self._q_lock.acquire()
        if frame_id in self._cached_data:
            del self._cached_data[frame_id]
        self._q_lock.release()

    def _add_q_dict(self, handle, frame_id):
        self._q_lock.acquire()
        if frame_id not in self._q_dict:
            self._q_dict[frame_id] = []
        self._q_dict[frame_id].append(handle)
        self._q_lock.release()

    def _abort_q(self, frame_id):
        self._q_lock.acquire()
        if frame_id in self._q_dict:
            for handle in self._q_dict[frame_id]:
                self._abort.add(handle)
            del self._q_dict[frame_id]
        self._q_lock.release()

    def _add_cache(self, data, frame_id, weak=False):
        self._q_lock.acquire()
        self._cached_data[frame_id] = [data,2-int(weak)]
        # self._overfill_cache(frame_id)
        self._q_lock.release()
    
    def _overfill_cache(self, frame_id):
        self._q_lock.acquire()
        if len(self._cached_data) <= self.CACHE_COUNT:
            self._q_lock.release()
            return
        start_idx = 0
        end_idx = len(self._temp_dirs) - 1
        prev_count = frame_id - start_idx
        next_count = end_idx - frame_id
        for i in range(max(prev_count, next_count),0,-1):
            if len(self._cached_data) <= self.CACHE_COUNT:
                break
            if frame_id - i >= start_idx and frame_id - i in self._cached_data :
                if self._cached_data[frame_id - i][1] == self.CacheType.WEAK:
                    self._cached_data[frame_id - i] = [self._temp_dirs[frame_id], self.CacheType.PATH]
            if frame_id + i <= end_idx and frame_id + i in self._cached_data:
                if self._cached_data[frame_id + i][1] == self.CacheType.WEAK:
                    self._cached_data[frame_id + i] = [self._temp_dirs[frame_id], self.CacheType.PATH]
        self._q_lock.release()

    def get_cache(self, frame_id):
        self._q_lock.acquire()
        if frame_id not in self._cached_data:
            self._q_lock.release()
            return [],[]
        if self._cached_data[frame_id][1] > self.CacheType.PATH:
            data = self._cached_data[frame_id][0]
            self._q_lock.release()
            return data
        else:
            data = self._load_with_preload(frame_id)
        self._q_lock.release()
        return data
    
    def _load(self,path):
        objs = []
        masks = []
        for obj in os.listdir(path):
            obj_path = os.path.join(path, obj)
            data = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            objs.append(int(os.path.splitext(obj)[0]))
            masks.append(data)
        return (objs, masks)


    def _load_with_preload(self, frame_id):
        self._q_lock.acquire()
        # load multi obj masks
        path = self._temp_dirs[frame_id]
        data = self._load(path)
        Thread(target=self._async_preload, args=(frame_id,), daemon=True).start()
        self._add_cache(data, frame_id, weak=True)
        self._q_lock.release()
        return data

    def _async_preload(self, frame_id, preload_range=3):
        self._q_lock.acquire()
        if frame_id in self._cached_data:
            self._q_lock.release()
            return
        if self._cached_data[frame_id][1] > self.CacheType.PATH:
            self._q_lock.release()
            return
        start_idx = max(0, frame_id - preload_range)
        end_idx = min(len(self._temp_dirs) - 1, frame_id + preload_range)
        prev_count = frame_id - start_idx
        next_count = end_idx - frame_id
        for i in range(max(prev_count, next_count)):
            if frame_id - i >= start_idx:
                if frame_id - i not in self._cached_data:
                    path = self._temp_dirs[frame_id - i]
                    # load multi obj masks
                    data = self._load(path)
                    self._add_cache(data, frame_id - i, weak=True)
            if frame_id + i <= end_idx:
                if frame_id + i not in self._cached_data:
                    path = self._temp_dirs[frame_id + i]
                    # load multi obj masks
                    data = self._load(path)
                    self._add_cache(data, frame_id + i, weak=True)
        self._q_lock.release()
  
        