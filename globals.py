class GlobalVariables:
    _shared_data = {}

    def __setattr__(self, key, value):
        self._shared_data[key] = value

    def __getattr__(self, key):
        if key in self._shared_data:
            return self._shared_data[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

# Use instance of GlobalVariables for sharing variables
global_vars = GlobalVariables()
