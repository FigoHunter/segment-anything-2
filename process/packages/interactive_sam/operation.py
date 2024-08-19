from enum import Enum
from .keyboard import Keys, Event

class Operation(Enum):
    NONE            = 0
    NEXT_IMG        = 1
    PREV_IMG        = 2
    NEXT_OBJ        = 3
    PREV_OBJ        = 4
    SAVE            = 5
    UNDO            = 6
    REDO            = 7
    PROCESS         = 8
    PROCESS_REVERT  = 9
    CLEAR           = 100
    QUIT            = 999

    def get_key(self):
        if self == Operation.NEXT_IMG:
            return Keys.RIGHT
        elif self == Operation.PREV_IMG:
            return Keys.LEFT
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
        elif self == Operation.PROCESS_REVERT:
            return Keys.CTRL_L | Keys.KEY_R
        elif self == Operation.CLEAR:
            return Keys.DELETE
        elif self == Operation.QUIT:
            return Keys.ESC

    def get_event(self):
        return Event.PRESS