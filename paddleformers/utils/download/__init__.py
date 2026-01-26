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

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "aistudio_hub_download": [
        "LocalTokenNotFoundError",
        "_clean_token",
        "_get_token_from_environment",
        "_get_token_from_file",
        "get_token",
        "get_token_to_send",
        "_validate_token_to_send",
        "build_aistudio_headers",
        "get_aistudio_file_metadata",
        "aistudio_hub_url",
        "aistudio_hub_download",
        "aistudio_hub_file_exists",
        "aistudio_hub_try_to_load_from_cache",
    ],
    "common": [
        "_cache_commit_hash_for_specific_revision",
        "_check_disk_space",
        "http_get",
        "_chmod_and_replace",
        "repo_folder_name",
        "OfflineModeIsEnabled",
        "OfflineAdapter",
        "_default_backend_factory",
        "_get_session_from_cache",
        "reset_sessions",
        "get_session",
        "_request_wrapper",
        "_get_pointer_path",
        "_create_symlink",
        "_set_write_permission_and_retry",
        "SoftTemporaryDirectory",
        "_to_local_dir",
        "_normalize_etag",
        "AistudioBosFileMetadata",
        "raise_for_status",
        "are_symlinks_supported",
    ],
    "download": [
        "DownloadSource",
        "register_model_group",
        "check_repo",
        "strtobool",
        "resolve_file_path",
        "hf_file_exist",
        "hf_try_to_load_from_cache",
    ],
}

if TYPE_CHECKING:
    from .download import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
