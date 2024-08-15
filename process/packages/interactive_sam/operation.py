from enum import Enum
from .keyboard import Keys, Event

class Operation(Enum):
    NONE            = 0
    NEXT_IMG        = 1
    PREV_IMG        = 2
    NEXT_OBJ        = 3
    PREV_OBJ        = 4
    SAVE            = 3
    UNDO            = 4
    REDO            = 5
    PROCESS         = 6
    CLEAR           = 100
    QUIT            = 999

    def get_key(self):
        if self == Operation.NEXT_IMG:
            return Keys.CTRL_L | Keys.RIGHT
        elif self == Operation.PREV_IMG:
            return Keys.CTRL_L | Keys.LEFT
        elif self == Operation.SAVE:
            return Keys.CTRL_L | Keys.ENTER
        elif self == Operation.UNDO:
            return Keys.CTRL_L | Keys.KEY_Z
        elif self == Operation.REDO:
            return Keys.CTRL_L | Keys.KEY_Y
        elif self == Operation.NEXT_OBJ:
            return Keys.CTRL_L | Keys.DOWN
        elif self == Operation.PREV_OBJ:
            return Keys.CTRL_L | Keys.UP
        elif self == Operation.PROCESS:
            return Keys.CTRL_L | Keys.KEY_P
        elif self == Operation.CLEAR:
            return Keys.DELETE
        elif self == Operation.QUIT:
            return Keys.ESC

    def get_event(self):
        return Event.PRESS