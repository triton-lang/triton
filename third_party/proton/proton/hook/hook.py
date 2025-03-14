from triton.compiler import LazyDict
from abc import abstractmethod
from typing import Dict, Any, Type, Optional
from collections import defaultdict
from triton.compiler.compiler import CompiledKernel


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
    # hook_type -> hook
    hooks: Dict[Type[Hook], Hook] = defaultdict(Hook)
    # session_id -> (hook_type -> active)
    sessions: Dict[int, Dict[Type[Hook], bool]] = defaultdict(lambda: defaultdict(lambda: False))

    @staticmethod
    def init_handle(function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        for hook in HookManager.hooks.values():
            hook.init_handle(function, module, metadata_group)

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        for hook in HookManager.hooks.values():
            hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        for hook in HookManager.hooks.values():
            hook.exit(lazy_dict)

    @staticmethod
    def _collect_hook_types(session: Optional[int] = None) -> set:
        hook_types = []
        if session:
            hook_types = HookManager.sessions[session]
        else:
            hook_types = set(hook_type for session_hooks in HookManager.sessions.values()
                             for hook_type in session_hooks)
        return hook_types

    @staticmethod
    def activate(session: Optional[int] = None) -> None:
        hook_types = HookManager._collect_hook_types(session)
        for hook_type in hook_types:
            HookManager.sessions[session][hook_type] = True
            if hook_type not in HookManager.hooks:
                HookManager.hooks[hook_type].activate()

    @staticmethod
    def deactivate(session: Optional[int] = None) -> None:
        hook_types = HookManager._collect_hook_types(session)
        for hook_type in hook_types:
            HookManager.sessions[session][hook_type] = False
            if not any(session_hooks[hook_type] for session_hooks in HookManager.sessions.values()):
                HookManager.hooks[hook_type].deactivate()

    @staticmethod
    def register(hook: Hook, session: int) -> None:
        HookManager.sessions[session][type(hook)] = True
        if type(hook) not in HookManager.hooks:
            hook.activate()
            HookManager.hooks[type(hook)] = hook
        CompiledKernel.launch_enter_hook = HookManager.enter
        CompiledKernel.launch_exit_hook = HookManager.exit
        CompiledKernel.init_handle_hook = HookManager.init_handle

    @staticmethod
    def unregister(session: Optional[int] = None) -> None:
        if session and session not in HookManager.sessions:
            return

        if not session:
            for hook in HookManager.hooks.values():
                hook.deactivate()
            HookManager.hooks.clear()
            HookManager.sessions.clear()
        else:
            hook_types = HookManager._collect_hook_types(session)
            # Deactivate hooks that are not used by any other session
            for hook_type in hook_types:
                if not any(session_hooks[hook_type] for session_hooks in HookManager.sessions.values()):
                    HookManager.hooks[hook_type].deactivate()
                    HookManager.sessions[session][hook_type] = False
                    HookManager.hooks.pop(hook_type)
        if not HookManager.hooks:
            CompiledKernel.launch_enter_hook = None
            CompiledKernel.launch_exit_hook = None
            CompiledKernel.init_handle_hook = None
