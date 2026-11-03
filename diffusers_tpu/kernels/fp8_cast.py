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

"""The fp8 cast the attention kernels apply to Q and K.

Kept here, free of any torchax import, so the Pallas kernels do not depend
on a runtime. Saturating rather than scaled: these tensors already sit
inside E4M3 range by construction. The scaled path, which needs a
calibrated scale, lives with the quantized linear layer.
"""

import jax.numpy as jnp

from . import e4m3


def quantize_saturating(x: "jnp.ndarray") -> "jnp.ndarray":
  """Casts a JAX array to E4M3, clamping instead of overflowing to NaN.

  Args:
    x: Real-dtype JAX array.

  Returns:
    ``jnp.float8_e4m3fn`` array with ``x``'s shape.
  """
  return jnp.clip(x.astype(jnp.float32), -e4m3.E4M3_MAX, e4m3.E4M3_MAX).astype(
      jnp.float8_e4m3fn
  )
