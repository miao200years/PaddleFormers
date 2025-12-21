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

# from numba.cuda.printimpl import print_item


class TorchBlocker:
    def __init__(self, verbose: bool = True):
        # self.verbose = verbose
        #
        # # 状态
        # self._stack = []
        self.torch_module = {}
        self.PF = 0
        self.PF_RESR = 1
        self.block_torch = False
        # 保存原始函数
        self._original_import = builtins.__import__
        self._original_find_spec = importlib.util.find_spec
        builtins.__import__ = self._custom_import
        importlib.util.find_spec = self._fake_find_spec

    def _fake_find_spec(self, name, package=None):
        """假的 find_spec，让 transformers 认为 torch 不存在"""

        # print(">>",filename)
        if self.block_torch and (name == "torch" or name.startswith("torch.")):
            return None
        return self._original_find_spec(name, package)

    def _is_called_from_paddleformers(self, current_globals) -> bool:
        """
        判断当前 import 是否出自 paddleformers 代码：
        - 先看调用方模块名（globals 的 __package__ / __name__）
        - 再回溯 Python 调用栈中的文件路径
        """
        # 1) 通过调用方模块名判断
        if current_globals is not None:
            caller_mod = current_globals.get("__package__") or current_globals.get("__name__", "") or ""
            if caller_mod.startswith("paddleformers"):
                return True

        # 2) 通过调用栈的文件路径判断
        # print("BG")
        for frame_info in traceback.extract_stack():
            filename = frame_info.filename or ""
            # print(">>",filename)
            if "paddleformers" in filename and "torch_blocker" not in filename:
                # print("END")
                return True
        # print("END")
        return False

    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """自定义 import 函数，只对 paddleformers / transformers / torch 生效"""
        # 计算完整模块名 full_name
        # print("name", name)
        for frame_info in traceback.extract_stack():
            filename = frame_info.filename or ""
            # print(f">>>>>>>{filename}:{frame_info.lineno}")
            if "PaddleFormers/tests/" in filename:
                return self._original_import(name, globals, locals, fromlist, level)
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

        # 快速路径：与我们无关的模块，完全不干预，避免影响 paddle / tests 等
        if top_level not in ("paddleformers", "transformers", "torch"):
            return self._original_import(name, globals, locals, fromlist, level)
        # if top_level in ("transformers", "torch"):
        #     self._stack.append(full_name)
        # print("<<<in", full_name)
        if top_level == "paddleformers":
            for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
                sys.modules.pop(module)

        # 如果当前处于 torch 屏蔽模式，禁止任何位置导入 torch

        # 判断当前调用是否来自 paddleformers

        from_paddleformers = self._is_called_from_paddleformers(globals)
        if from_paddleformers is False:
            if self.PF is False:
                self.block_torch = False
                # importlib.util.find_spec = self._original_find_spec
                # return self._original_import(name, globals, locals, fromlist, level)
            else:
                for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
                    sys.modules.pop(module)
                    # print("pop:",module)
                for module_name, module in self.torch_module.items():
                    sys.modules[module_name] = module
                self.torch_module = {}

                # while self._stack:
                #     module = self._stack.pop()
                #     try:
                #
                #     except:
                #         pass
                self.PF = False
                self.PF_RESR = True
                self.block_torch = False
                # importlib.util.find_spec = self._original_find_spec
        else:
            self.PF = True
            if self.PF_RESR is True:

                for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
                    sys.modules.pop(module)
                    # print("pop:", module)
                for module_name in [i for i in sys.modules.keys() if i.startswith("torch")]:
                    module = sys.modules.pop(module_name)
                    self.torch_module[module_name] = module

                self.PF_RESR = False
                self.block_torch = True
                # importlib.util.find_spec = self._fake_find_spec
            else:
                self.block_torch = True
                # importlib.util.find_spec = self._fake_find_spec

        return self._original_import(name, globals, locals, fromlist, level)
