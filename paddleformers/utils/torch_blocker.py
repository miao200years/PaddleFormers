# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import importlib.util
import sys
import traceback


class TorchBlocker:
    def __init__(self, verbose: bool = True):

        self.verbose = verbose

        self._stack = []
        self._torch_blocked = False
        self._torch_backup = {}
        self._transformers_by_paddleformers = False

        self._original_import = builtins.__import__
        self._original_find_spec = importlib.util.find_spec

        self._start()

    def _log(self, msg: str):

        if self.verbose:
            print(f"[TorchBlocker] {msg}")

    def _fake_find_spec(self, name, package=None):

        if self._torch_blocked and (name == "torch" or name.startswith("torch.")):
            return None
        return self._original_find_spec(name, package)

    def _block_torch(self):

        if self._torch_blocked:
            return

        self._torch_backup = {}
        for name in list(sys.modules.keys()):
            if name == "torch" or name.startswith("torch."):
                self._torch_backup[name] = sys.modules.pop(name)

        importlib.util.find_spec = self._fake_find_spec
        self._torch_blocked = True
        self._log("torch 已屏蔽")

    def _unblock_torch(self):

        if not self._torch_blocked:
            return

        importlib.util.find_spec = self._original_find_spec

        for name, mod in self._torch_backup.items():
            sys.modules[name] = mod
        self._torch_backup = {}

        self._torch_blocked = False
        self._log("torch 已恢复")

    def _clear_transformers_cache(self):

        to_remove = [
            name for name in list(sys.modules.keys()) if name == "transformers" or name.startswith("transformers.")
        ]
        for name in to_remove:
            del sys.modules[name]
        if to_remove:
            self._log(f"已清除 {len(to_remove)} 个 transformers 模块缓存")

    def _is_called_from_paddleformers(self, current_globals) -> bool:

        if current_globals is not None:
            caller_mod = current_globals.get("__package__") or current_globals.get("__name__", "") or ""
            if caller_mod.startswith("paddleformers"):
                return True

        for frame_info in traceback.extract_stack():
            filename = frame_info.filename or ""
            if "paddleformers" in filename and "torch_blocker" not in filename:
                return True

        return False

    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):

        if level > 0 and globals:
            pkg = globals.get("__package__") or globals.get("__name__", "")
            if pkg:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - level + 1]) if level <= len(parts) else ""
                full_name = f"{base}.{name}" if name and base else (name or base)
            else:
                full_name = name
        else:
            full_name = name

        top_level = (full_name or "").split(".")[0]

        if top_level not in ("paddleformers", "transformers", "torch"):
            return self._original_import(name, globals, locals, fromlist, level)

        if self._torch_blocked and top_level == "torch":
            raise ImportError("torch is blocked (paddleformers mode)")

        from_paddleformers = self._is_called_from_paddleformers(globals)

        if top_level == "transformers" and from_paddleformers:

            if not self._torch_blocked:
                self._block_torch()
            self._transformers_by_paddleformers = True
            self._log("transformers 被 paddleformers 导入 (torch 屏蔽中)")

        elif top_level == "transformers" and not from_paddleformers:
            if self._transformers_by_paddleformers or self._torch_blocked:

                self._log("用户直接 import transformers，切换为正常模式")
                self._unblock_torch()
                self._clear_transformers_cache()
                self._transformers_by_paddleformers = False

        self._stack.append(full_name)
        try:
            return self._original_import(name, globals, locals, fromlist, level)
        finally:
            self._stack.pop()

    def _start(self):

        builtins.__import__ = self._custom_import

    def stop(self):

        builtins.__import__ = self._original_import
        self._unblock_torch()
        self._log("已停止")

    def reset(self):

        self._stack = []
        self._transformers_by_paddleformers = False
        self._unblock_torch()

    @property
    def is_torch_blocked(self) -> bool:
        return self._torch_blocked

    @property
    def is_transformers_loaded_by_paddleformers(self) -> bool:
        return self._transformers_by_paddleformers
