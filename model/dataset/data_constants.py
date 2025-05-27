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

# The input image size is selected with several factors:
# - With the StyleGan decoder, the output RGB/semantic image size is calculated as
#   64 * StyleGanDecoder.constant_size, where constant_size is currently set as (5, 8).
#   While not stricly required, it's better to make the input image as the same size
#   for easier evaluation and analysis.
# - Trade-off between GPU memory usage and model performace. X-mobility is trained with sequences
#   of frames, requiring large amount of GPU memory, therefore it's preferred to keeping the
#   input image size relatively small to increase the sequence_length for better model performance.
INPUT_IMAGE_SIZE = (320, 512)
