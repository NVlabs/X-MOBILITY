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
import numpy as np
from enum import IntEnum

LEROBOT_SEMANTIC_COLORS = np.array(
    [
        [0, 0, 0],  # Background
        [255, 255, 255],  # NavigableSurface
        [255, 0, 0],  # Forklift
        [0, 255, 0],  # Pallet
        [0, 0, 255],  # Cone
        [255,255,0],  # Sign
        [255,0,255],
        [192,192,192],
        [128,128,128],
        [128,0,0],
        [128,128,0],# Fence
        [0,128,0],
        [128,0,128],
        [0,128,128],
        [0,0,128],
    ],
    dtype=np.uint8)

class LeRobotSemanticLabel(IntEnum):
    ''' Semantic label enum
    '''
    BACKGROUND = 0
    UNLABELLED = 1
    CONE = 2
    CRATE = 3
    FENCE = 4
    FLOOR = 5
    FLOOR_STRUCTURE = 6
    FORKLIFT = 7
    SIGN = 8
    LINE = 9
    PALLET_SHELF = 10
    PILE = 11
    RACK_SHELF = 12
    SHELF = 13
    STRUCTURE = 14

    @staticmethod
    def get_semantic_lable_names():
        return [label.name for label in LeRobotSemanticLabel]
