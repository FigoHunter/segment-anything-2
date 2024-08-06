import copy
import uuid


class HistorySlice:
    def __init__(self, maxsize = None) -> None:
        self.__list= []
        self.__start_index=0
        self.__size = 0
        self.__maxsize = maxsize

    def __len__(self):
        return self.__size

    def __get_real_index(self, index):
        if index >= self.__size:
            raise IndexError
        if index < 0:
            index += self.__size
        if index < 0:
            raise IndexError
        index = self.__start_index + index
        if self.__maxsize is not None and index >= self.__maxsize:
            index -= self.__maxsize
        # if index >= self.__maxsize:
        #     index -= self.__maxsize
        return index

    def __getitem__(self, index):
        index = self.__get_real_index(index)
        return self.__list[index]

    def __setitem__(self, index, value):
        index = self.__get_real_index(index)
        self.__list[index]=value

    def append(self, value):
        if self.__maxsize is not None and self.__size >= self.__maxsize:
            self.__start_index += 1
            self.__size -= 1
        self.__size += 1
        index = self.__get_real_index(self.__size - 1)
        if index >= len(self.__list):
            self.__list.append(value)
        else:
            self.__list[index] = value

    def resize(self, size):
        if size < 0:
            raise ValueError
        if size > self.__size:
            size = self.__size
            print("Warning: size is too large, resizing to {}".format(size))
            return
        self.__size = size

    def __iter__(self):
        for i in range(self.__size):
            yield self.__getitem__(i)
    
    def __str__(self) -> str:
        ls = list(self)
        return f'[{", ".join(str(x) for x in ls)}]'

class History:
    # @classmethod
    # def instance(cls):
    #     if not hasattr(cls, "_instance"):
    #         cls._instance = cls()
    #     return cls._instance

    def __init__(self,obj,maxsize=None) -> None:
        self.__history = HistorySlice(maxsize)
        self.__history_index = -1
        self.register(obj)
        # self.__history = {}
        # self.__history_index = {}

    # def __key(self, obj):
    #     if hasattr(obj, '__history_id'):
    #         return obj.__history_id
    #     else:
    #         obj.__history_id = uuid.uuid4().hex.replace('-','')
        
    def register(self,obj):
        # history = self.__get_history(obj)
        # index = self.__get_index(obj)
        history = self.__history
        index = self.__history_index
        if index < len(history) - 1:
            self.__trim_history(index)
        self.__history_index = len(history)
        history.append(copy.deepcopy(obj))


    def undo(self, error_if_nothing_to_undo=True):
        index = self.__history_index
        if index > 0:
            index -= 1
            self.__history_index = index
            return copy.deepcopy(self.__history[index])
        elif error_if_nothing_to_undo:
            raise ValueError("Nothing to undo")
        else:
            return None
        
    def redo(self, error_if_nothing_to_redo=True):
        index = self.__history_index
        if index < len(self.__history) - 1:
            index += 1
            self.__history_index = index
            return copy.deepcopy(self.__history[index])
        elif error_if_nothing_to_redo:
            raise ValueError("Nothing to redo")
        else:
            return None

    # def __get_history(self,obj):
    #     if self.__key(obj) not in self.__history:
    #         self.__history[self.__key(obj)] = HistorySlice(MAX_HISTORY_SIZE)
    #     return self.__history[self.__key(obj)]
    
    # def __get_index(self,obj):
    #     if self.__key(obj) not in self.__history_index:
    #         self.__history_index[self.__key(obj)] = -1
    #     return self.__history_index[self.__key(obj)]
    
    def __trim_history(self,index):
        history = self.__history
        history.resize(index+1)
