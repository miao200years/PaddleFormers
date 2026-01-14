# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


def compare_version(version, pair_version):
    """
    Args:
        version (str): The first version string needed to be compared.
            The format of version string should be as follow : "xxx.yyy.zzz".
        pair_version (str): The second version string needed to be compared.
             The format of version string should be as follow : "xxx.yyy.zzz".
    Returns:
        int: The result of comparison. 1 means version > pair_version; 0 means
            version = pair_version; -1 means version < pair_version.

    Examples:
        >>> compare_version("2.2.1", "2.2.0")
        >>> 1
        >>> compare_version("2.2.0", "2.2.0")
        >>> 0
        >>> compare_version("2.2.0-rc0", "2.2.0")
        >>> -1
        >>> compare_version("2.3.0-rc0", "2.2.0")
        >>> 1
    """
    version = version.strip()
    pair_version = pair_version.strip()
    if version == pair_version:
        return 0
    version_list = version.split(".")
    pair_version_list = pair_version.split(".")
    for version_code, pair_version_code in zip(version_list, pair_version_list):
        if not version_code.isnumeric():
            return -1
        if not pair_version_code.isnumeric():
            return 1
        if int(version_code) > int(pair_version_code):
            return 1
        elif int(version_code) < int(pair_version_code):
            return -1
    return 0


import importlib
import warnings


def _check_dependency_versions():
    for pkg_name, min_version in [("paddlepaddle", "3.3"), ("paddlefleet", "0.1")]:
        version = importlib.metadata.version(pkg_name)
        if compare_version(version, min_version) < 0:
            warnings.warn("Version check warning:\n" + f"{pkg_name} version {version}, recommended >= {min_version}")


_check_dependency_versions()
