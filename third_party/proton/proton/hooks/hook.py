from triton.compiler import LazyDict
from abc import abstractmethod
from typing import Dict, Any, Optional
from collections import defaultdict
import triton.knobs as knobs


class Hook:
    priority: int = 0

    @abstractmethod
    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str],
                    hash: str) -> None:  # noqa: D401
        raise NotImplementedError

    @abstractmethod
    def enter(self, metadata: LazyDict) -> None:
        raise NotImplementedError

    @abstractmethod
    def exit(self, metadata: LazyDict) -> None:
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
    def init_handle(module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        for hook in HookManager.active_hooks:
            hook.init_handle(module, function, name, metadata_group, hash)

    @staticmethod
    def enter(metadata: LazyDict) -> None:
        for hook in HookManager.active_hooks:
            hook.enter(metadata)

    @staticmethod
    def exit(metadata: LazyDict) -> None:
        # It's important to reverse the order of hooks so that we keep the first in last out order
        for hook in reversed(HookManager.active_hooks):
            hook.exit(metadata)

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
                if hook in HookManager.active_hooks:
                    deactivated_hooks.add(hook)
                HookManager.session_hooks[session][hook] = False

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
        knobs.runtime.kernel_load_end_hook.add(HookManager.init_handle)
        knobs.runtime.launch_enter_hook.add(HookManager.enter)
        knobs.runtime.launch_exit_hook.add(HookManager.exit)

    @staticmethod
    def unregister(session: Optional[int] = None) -> None:
        if session is not None and session not in HookManager.session_hooks:
            return

        if session is None:
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
            knobs.runtime.kernel_load_end_hook.remove(HookManager.init_handle)
            knobs.runtime.launch_enter_hook.remove(HookManager.enter)
            knobs.runtime.launch_exit_hook.remove(HookManager.exit)
