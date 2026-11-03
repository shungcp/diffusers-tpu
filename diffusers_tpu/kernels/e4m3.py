# Copyright 2026 Google LLC
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

"""Constants of the E4M3 float8 encoding.

The representable range of ``float8_e4m3fn`` and the floor a computed scale
is clamped to. Shared by the quantized linear layer and the fp8 cast the
attention kernels apply to Q and K; this module has no imports so either
side can use it.
"""

# Largest magnitude ``jnp.float8_e4m3fn`` / ``torch.float8_e4m3fn``
# represents. Values beyond it saturate rather than overflow to NaN.
E4M3_MAX: float = 448.0

# Floor for a computed scale, so an all-zero tensor cannot divide by zero.
EPS: float = 1e-5
