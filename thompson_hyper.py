
import sys

import numpy as np

import mahh as B


class ThompsonLearner:
    """Thompson sampling (Sección 4.2.2)."""
    NAME = "Thompson"

    def __init__(self, H, rng):
        self.H, self.rng = H, rng
        self.P = np.ones((H, H))
        self.N = 2 * np.ones((H, H))
        self.Q = np.ones(H)
        self.M = 2 * np.ones(H)

    def sample_next(self, i_last):
        theta = self.rng.beta(self.P[i_last], np.maximum(self.N[i_last] - self.P[i_last], 1e-9))
        return int(np.argmax(theta))

    def sample_apply(self, i_next):
        q, m = self.Q[i_next], self.M[i_next]
        phi_0 = self.rng.beta(m - q, q)
        phi_1 = self.rng.beta(q, m - q)
        return 1 if phi_1 > phi_0 else 0

    def update(self, sequence, improved):
        if not improved or len(sequence) < 1:
            return
        n = len(sequence)
        for k in range(n):
            ik = sequence[k]
            self.N[ik, :] += 1
            self.M[ik] += 1
            if k < n - 1:
                self.P[ik, sequence[k + 1]] += 1
        self.Q[sequence[-1]] += 1


def main():
    ap = B.base_parser("JVCRSP: MAHH con Thompson sampling")
    return B.run_cli(ThompsonLearner, ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())