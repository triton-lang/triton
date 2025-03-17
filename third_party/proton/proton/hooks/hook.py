from triton.compiler import LazyDict
from abc import abstractmethod
from typing import Dict, Any, Optional, Set
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
    # active hooks
    active_hooks: Set[Hook] = set()
    # session_id -> (hook_type -> active)
    session_hooks: Dict[int, Dict[Hook, bool]] = defaultdict(lambda: defaultdict(bool))

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
    def activate(session: Optional[int] = None) -> None:
        for hook in HookManager.session_hooks[session]:
            if hook not in HookManager.active_hooks:
                hook.activate()
                HookManager.active_hooks.add(hook)
            HookManager.session_hooks[session][hook] = True

    @staticmethod
    def deactivate(session: Optional[int] = None) -> None:
        for hook in HookManager.session_hooks[session]:
            HookManager.session_hooks[session][hook] = False
            # Check if any other sessions rely on this hook
            if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                hook.deactivate()
                HookManager.active_hooks.remove(hook)

    @staticmethod
    def register(hook: Hook, session: int) -> None:
        HookManager.session_hooks[session][hook] = True
        if hook not in HookManager.active_hooks:
            hook.activate()
            HookManager.active_hooks.add(hook)
        if CompiledKernel.launch_enter_hook is None:
            CompiledKernel.launch_enter_hook = HookManager.enter
            CompiledKernel.launch_exit_hook = HookManager.exit
            CompiledKernel.init_handle_hook = HookManager.init_handle

    @staticmethod
    def unregister(session: Optional[int] = None) -> None:
        if session and session not in HookManager.session_hooks:
            return

        if not session:
            for hook in HookManager.active_hooks:
                hook.deactivate()
            HookManager.active_hooks.clear()
            HookManager.session_hooks.clear()
        else:
            popped_hooks = HookManager.session_hooks.pop(session)
            # Deactivate hooks that are not used by any other session
            for hook, active in popped_hooks.items():
                if not active:
                    continue
                if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                    hook.deactivate()
                    HookManager.active_hooks.remove(hook)
        if not HookManager.active_hooks:
            CompiledKernel.launch_enter_hook = None
            CompiledKernel.launch_exit_hook = None
            CompiledKernel.init_handle_hook = None
