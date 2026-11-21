# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run a callback right after a named module finishes importing.

The AReaL worker entry point installs patches lazily: a rollout worker never
imports ``areal.engine.megatron_engine``, so Megatron is never imported there.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable
from types import ModuleType

Callback = Callable[[ModuleType], None]


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, callbacks: list[Callback]) -> None:
        self._wrapped = wrapped
        self._callbacks = callbacks

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return self._wrapped.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped.exec_module(module)
        for callback in self._callbacks:
            callback(module)

    def __getattr__(self, name: str):  # get_source, get_filename, is_package, ...
        return getattr(self._wrapped, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.callbacks: dict[str, list[Callback]] = {}
        self._resolving: set[str] = set()

    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        callbacks = self.callbacks.get(fullname)
        if not callbacks or fullname in self._resolving:
            return None
        self._resolving.add(fullname)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None:
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        break
        finally:
            self._resolving.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _PostImportLoader(spec.loader, callbacks)
        return spec


_FINDER: _PostImportFinder | None = None


def when_imported(module_name: str, callback: Callback) -> None:
    """Call ``callback(module)`` now if imported, otherwise right after import."""

    module = sys.modules.get(module_name)
    if module is not None:
        callback(module)
        return
    global _FINDER
    if _FINDER is None:
        _FINDER = _PostImportFinder()
        sys.meta_path.insert(0, _FINDER)
    _FINDER.callbacks.setdefault(module_name, []).append(callback)


__all__ = ["when_imported"]
