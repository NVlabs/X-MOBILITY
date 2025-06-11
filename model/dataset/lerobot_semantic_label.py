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


class LeRobotSemanticLabel(IntEnum):
    ''' Semantic label enum
    '''
    BACKGROUND = 0
    UNLABELLED = 1
    FORKLIFT = 2
    PALLET = 3
    CONE = 4
    SIGN = 5
    FENCE = 6
    CRATE = 7
    FLOOR = 8
    STRUCTURE = 9
    LINE = 10
    PILE = 11
    SHELF = 12
    RACK = 13
    NAVIGABLE = 14

    @staticmethod
    def get_semantic_lable_names():
        return [label.name for label in LeRobotSemanticLabel]
