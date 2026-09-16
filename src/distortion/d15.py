"""Low-order ablation: Noll j=2..10, with no fitted residual Gaussian.

D15 and D2 share the pupil, covariance, optical scale, truncation window and
registration convention. D15 deliberately omits modes j>10. Consequently it
need not have the same ensemble blur strength as D2. It is NOT an exact
conditional expectation over omitted modes, nor a published Chan level.
"""
from .d2 import D2Kolmogorov
from .wavefront import K_MODES

J_MAX = 10

class D15LowOrder(D2Kolmogorov):
    name = "d15"

    def __init__(self, diffraction_fwhm_px, j_max=J_MAX):
        super().__init__(diffraction_fwhm_px)
        self.j_max = int(j_max)
        if not 3 <= self.j_max <= K_MODES:
            raise ValueError("j_max must be between 3 and K_MODES")

    def __call__(self, img, d_over_r0, rng):
        a = self.wf.coeffs(d_over_r0, rng)
        a[self.j_max - 1:] = 0.0
        return self._render_coeffs(img, a)
