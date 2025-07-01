# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import IntEnum
import numpy as np

SEMANTIC_COLORS = np.array(
    [
        [128, 128, 128],  # Background
        [0, 255, 0],  # NavigableSurface
        [255, 165, 0],  # Forklift
        [0, 0, 255],  # Pallet
        [255, 255, 0],  # Cone
        [255, 0, 255],  # Sign
        [255, 0, 0]  # Fence
    ],
    dtype=np.uint8)

class SemanticLabel(IntEnum):
    ''' Semantic label enum
    '''
    BACKGROUND = 0
    NAVIGABLE = 1
    FORKLIFT = 2
    PALLET = 3
    CONE = 4
    SIGN = 5
    FENCE = 6

    @staticmethod
    def get_semantic_lable_names():
        return [label.name for label in SemanticLabel]