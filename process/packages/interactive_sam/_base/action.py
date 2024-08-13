class Action:
    def __init__(self):
        self._actions = []

    def __iadd__(self, action):
        """Support for the += operator to add an action"""
        self._actions.append(action)
        return self

    def __isub__(self, action):
        """Support for the -= operator to remove an action"""
        self._actions.remove(action)
        return self

    def trigger(self, *args, **kwargs):
        """Invoke all registered actions"""
        for action in self._actions:
            action(*args, **kwargs)
