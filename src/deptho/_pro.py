"""Optional hook into the closed-source ``deptho_pro`` extension."""
from __future__ import annotations

import importlib


class _Pro:
    _checked = False
    _module = None

    def _check(self):
        if not self._checked:
            self._checked = True
            try:
                mod = importlib.import_module("deptho_pro")
                if mod.license_status().valid:
                    self._module = mod
            except Exception:
                self._module = None

    @property
    def available(self) -> bool:
        self._check()
        return self._module is not None

    @property
    def module(self):
        self._check()
        return self._module

    def reset(self):
        self._checked, self._module = False, None


pro = _Pro()
