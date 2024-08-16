class LazyLoader:
    def __init__(self, import_path, class_name):
        self.import_path = import_path
        self.class_name = class_name
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            module = __import__(self.import_path, fromlist=[self.class_name])
            class_ = getattr(module, self.class_name)
            self._instance = class_(*args, **kwargs)
        return self._instance
