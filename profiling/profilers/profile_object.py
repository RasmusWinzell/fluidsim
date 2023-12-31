class ProfileObject:
    def __init__(
        self,
        obj,
        code,
        start_line,
        module,
        profiler=None,
        profile_code=None,
        profile_obj=None,
    ):
        self._obj = obj
        self._code = code
        self._start_line = start_line
        self._module = module
        self._profiler = profiler
        self._profile_code = profile_code
        self._profile_obj = profile_obj

    @property
    def obj(self):
        return self._obj

    @property
    def code(self):
        return self._code

    @property
    def start_line(self):
        return self._start_line

    @property
    def module(self):
        return self._module

    @property
    def profiler(self):
        return self._profiler

    @profiler.setter
    def profiler(self, profiler):
        self._profiler = profiler

    @property
    def profile_code(self):
        return self._profile_code

    @profile_code.setter
    def profile_code(self, profile_code):
        self._profile_code = profile_code

    @property
    def profile_obj(self):
        return self._profile_obj

    @profile_obj.setter
    def profile_obj(self, profile_obj):
        self._profile_obj = profile_obj

    @property
    def is_module(self):
        return self.obj == self.module

    @property
    def name(self):
        if self.is_module:
            return self.obj.__name__
        return f"{self.module.__name__}.{self.obj.__name__}"
