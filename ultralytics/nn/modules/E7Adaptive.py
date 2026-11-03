# -*- coding: utf-8 -*-
"""E7 independent module; does not replace E6 MLLA."""

# ===== MEND_E7_ADAPTIVE_BEGIN =====
from ultralytics.nn.Addmodules.MLLABlock import C2fMLLAHybrid
import torch
import torch.nn as nn

class C2fMLLAAdaptive(C2fMLLAHybrid):
    """E7: E6-equivalent initial residual + spatial/channel adaptive gate."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, gamma_init=0.05):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e,
                         gamma_init=gamma_init)
        self.e7_dw = nn.Conv2d(self.c, self.c, 3, padding=1,
                               groups=self.c, bias=False)
        self.e7_act = nn.SiLU()
        self.e7_pw = nn.Conv2d(self.c, self.c, 1, bias=True)
        nn.init.zeros_(self.e7_pw.weight)
        nn.init.zeros_(self.e7_pw.bias)

    def _project_with_attention(self, parts):
        last = parts[-1]
        adaptive_gain = 2.0 * torch.sigmoid(
            self.e7_pw(self.e7_act(self.e7_dw(last))))
        delta = self.mlla(last) - last
        parts[-1] = last + torch.tanh(self.gamma) * adaptive_gain * delta
        return self.cv2(torch.cat(parts, dim=1))
# ===== MEND_E7_ADAPTIVE_END =====
