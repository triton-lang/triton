from triton.compiler import LazyDict
from abc import abstractmethod
from typing import Dict, Any, Optional
from collections import defaultdict
import triton.knobs as knobs


class Hook:
    priority: int = 0

    @abstractmethod
    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str]) -> None:
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
    active_hooks: list[Hook] = []
    # session_id -> (hook_type -> active)
    session_hooks: Dict[int, Dict[Hook, bool]] = defaultdict(lambda: defaultdict(bool))

    @staticmethod
    def init_handle(module: Any, function: Any, name: str, metadata_group: Dict[str, str]) -> None:
        for hook in HookManager.active_hooks:
            hook.init_handle(module, function, name, metadata_group)

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        for hook in HookManager.active_hooks:
            hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        # It's important to reverse the order of hooks so that we keep the first in last out order
        for hook in reversed(HookManager.active_hooks):
            hook.exit(lazy_dict)

    @staticmethod
    def activate(session: Optional[int] = None) -> None:
        if session is None:
            sessions = HookManager.session_hooks.keys()
        else:
            sessions = [session]

        for session in sessions:
            for hook in HookManager.session_hooks[session]:
                if hook not in HookManager.active_hooks:
                    hook.activate()
                    HookManager.active_hooks.append(hook)
                HookManager.session_hooks[session][hook] = True
        # Sort active_hooks by priority
        HookManager.active_hooks.sort(key=lambda x: x.priority, reverse=True)

    @staticmethod
    def deactivate(session: Optional[int] = None) -> None:
        if session is None:
            sessions = HookManager.session_hooks.keys()
        else:
            sessions = [session]

        deactivated_hooks = set()
        for session in sessions:
            for hook in HookManager.session_hooks[session]:
                HookManager.session_hooks[session][hook] = False
                deactivated_hooks.add(hook)

        # Check if any other sessions rely on this hook
        for hook in deactivated_hooks:
            if not any(session_hooks[hook] for session_hooks in HookManager.session_hooks.values()):
                hook.deactivate()
                HookManager.active_hooks.remove(hook)

    @staticmethod
    def register(hook: Hook, session: int) -> None:
        HookManager.session_hooks[session][hook] = True
        if hook not in HookManager.active_hooks:
            hook.activate()
            HookManager.active_hooks.append(hook)
        # Sort active_hooks by priority
        HookManager.active_hooks.sort(key=lambda x: x.priority, reverse=True)
        # Register the heads
        if knobs.runtime.launch_enter_hook is None:
            knobs.runtime.launch_enter_hook = HookManager.enter
            knobs.runtime.launch_exit_hook = HookManager.exit
            knobs.runtime.init_handle_hook = HookManager.init_handle

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
        # Unregister the heads
        if not HookManager.active_hooks:
            knobs.runtime.launch_enter_hook = None
            knobs.runtime.launch_exit_hook = None
            knobs.runtime.init_handle_hook = None
