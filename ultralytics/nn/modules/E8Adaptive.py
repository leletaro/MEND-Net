# -*- coding: utf-8 -*-
"""E8 independent module; do not edit E7 source."""

# ===== MEND_E8_BETA10_BEGIN =====
from ultralytics.nn.modules.E7Adaptive import C2fMLLAAdaptive
import torch
import torch.nn as nn

class C2fMLLAAdaptiveBeta10(C2fMLLAAdaptive):
    """E8: E7 with only fixed logit response scale beta=10; no new tensors."""
    e8_beta = 10.0

    def _project_with_attention(self, parts):
        last = parts[-1]
        logits = self.e7_pw(self.e7_act(self.e7_dw(last)))
        adaptive_gain = 2.0 * torch.sigmoid(self.e8_beta * logits)
        delta = self.mlla(last) - last
        parts[-1] = last + torch.tanh(self.gamma) * adaptive_gain * delta
        return self.cv2(torch.cat(parts, dim=1))
# ===== MEND_E8_BETA10_END =====
