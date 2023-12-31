import inspect
from copy import copy
from typing import Any

from .profile_code import ProfileCode
from .profile_object import ProfileObject
from .profiler import Profiler


class AdvancedProfiler(Profiler):
    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self._objs = {}

    @property
    def profilers(self):
        return [copy(self)] + [pobj.profiler for pobj in self._objs.values()]

    def profile(self, obj: Any):
        pobj = ProfileObject(obj, *self._get_obj_source(obj))
        pobj.profiler = Profiler(self._format_profiler_name(pobj))
        pobj.profile_code = ProfileCode(pobj.code, pobj.start_line).__str__()

        print(pobj.profile_code)

        global_d = self._get_globals(pobj)
        local_d = self._exec_code(pobj.profile_code, global_d)
        self._set_profile_obj(pobj, local_d)

        self._objs[pobj.name] = pobj
        return pobj.profile_obj

    def _format_profiler_name(self, pobj: ProfileObject):
        if self.name is None:
            return None
        return f"{self.name}:{pobj.name}"

    def _exec_code(self, code: str, global_d: dict) -> Any:
        local_d = {}
        exec(code, global_d, local_d)
        return local_d

    @staticmethod
    def _set_profile_obj(pobj: ProfileObject, local_d: dict):
        if pobj.is_module:
            pobj.module.__dict__.update(local_d)
            pobj.profile_obj = pobj.module
        else:
            pobj.profile_obj = local_d[pobj.obj.__name__]

    @staticmethod
    def _get_globals(pobj: ProfileObject):
        global_d = {
            ProfileCode.PROFILER_VAR_NAME: pobj.profiler,
            AdvancedProfiler.SOURCE_VAR_NAME: pobj.code,
            AdvancedProfiler.SOURCE_START_VAR_NAME: pobj.start_line,
        }
        global_d.update(pobj.module.__dict__)
        return global_d

    @staticmethod
    def _get_obj_source(obj: Any):
        module = inspect.getmodule(obj)
        code = inspect.getsource(module)
        code_to_consider = inspect.getsource(obj)
        start_line = AdvancedProfiler._find_start_line(code_to_consider, code)
        return code_to_consider, start_line, module

    @staticmethod
    def _find_start_line(code, all_code):
        idx = all_code.find(code)
        if idx == -1:
            raise Exception("Code not found in all_code")
        return all_code[:idx].count("\n") + 1
