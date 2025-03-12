from triton.compiler import LazyDict
from abc import abstractmethod
from typing import List, Dict, Any
from collections import defaultdict
from triton._C.libproton import proton as libproton


class Hook:

    @abstractmethod
    def init_handle(self, function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def enter(self, lazy_dict: LazyDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def exit(self, lazy_dict: LazyDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def activate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def deactivate(self) -> None:
        raise NotImplementedError


class HookManager:
    hooks: Dict[int, List[Hook]] = defaultdict(list)
    active_hooks: List[Hook] = []

    @staticmethod
    def init_handle(function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        for hook in HookManager.active_hooks:
            hook.init_handle(function, module, metadata_group)

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        for hook in HookManager.active_hooks:
            hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        for hook in HookManager.active_hooks:
            hook.exit(lazy_dict)

    @staticmethod
    def register(session, hooks: List) -> None:
        HookManager.hooks[session].extend(hooks)
        HookManager.activate(session, hooks)

    @staticmethod
    def unregister(session) -> None:
        HookManager.hooks.pop(session)
        HookManager.deactivate(session)

    @staticmethod
    def activate(session, hooks: List) -> None:
        if HookManager.active_hooks:
            raise RuntimeError("Cannot activate hook while other hooks are active.")

        if session not in HookManager.hooks:
            return

        if libproton.get_num_active_sessions() > 0:
            raise RuntimeError("Cannot activate hook while other sessions are active.")

        HookManager.active_hooks.extend(hooks)
        for hook in hooks:
            hook.activate()

    @staticmethod
    def deactivate(session) -> None:
        if session not in HookManager.hooks:
            return

        for hook in HookManager.active_hooks:
            hook.deactivate()
        HookManager.active_hooks.clear()
