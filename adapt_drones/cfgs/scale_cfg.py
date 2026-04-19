from dataclasses import dataclass
import numpy as np


@dataclass
class Scale:
    scale: bool
    scale_lengths: list

    def __init__(self, scale, scale_lengths=None):
        self.scale = scale
        self.scale_lengths = scale_lengths
        # mass fit
        self.avg_mass_fit = np.array([1.2498e02, -1.0555e01, 2.8744e00, -1.0497e-01])
        self.std_mass_fit = np.array([-1.2207e02, 4.6697e01, -3.3219e00, 7.1909e-02])

        # ixx fit
        self.avg_ixx_fit = np.array(
            [5.5211e01, -1.5131e01, 2.7314e00, -2.0660e-01, 7.1531e-03, -9.2833e-05]
        )
        self.std_ixx_fit = np.array(
            [2.7843e01, -9.8623e00, 2.2186e00, -2.2685e-01, 1.0597e-02, -1.8536e-04]
        )

        # iyy fit
        self.avg_iyy_fit = np.array(
            [5.5211e01, -1.5131e01, 2.7314e00, -2.0660e-01, 7.1531e-03, -9.2833e-05]
        )
        self.std_iyy_fit = np.array(
            [2.7843e01, -9.8623e00, 2.2186e00, -2.2685e-01, 1.0597e-02, -1.8536e-04]
        )

        # izz fit
        self.avg_izz_fit = np.array(
            [1.1543e02, -3.7899e01, 6.8413e00, -5.1748e-01, 1.7916e-02, -2.3252e-04]
        )
        self.std_izz_fit = np.array(
            [8.7295e01, -4.2189e01, 9.2968e00, -9.1877e-01, 4.1948e-02, -7.2329e-04]
        )

        # km_kf fit
        self.avg_km_kf_fit = np.array([1.4044e-01, -5.6717e-03])
        self.std_km_kf_fit = np.array([3.3319e-02, 4.8779e-04])

        self._precompute_arm_length_pmf()

    def _precompute_arm_length_pmf(self, n_grid=1000):
        if (
            not self.scale
            or self.scale_lengths is None
            or self.scale_lengths[0] >= self.scale_lengths[1]
        ):
            self.arm_length_grid = None
            self.arm_length_pmf = None
            return

        L_grid = np.linspace(self.scale_lengths[0], self.scale_lengths[1], n_grid)
        volumes = np.empty(n_grid)

        for i, L in enumerate(L_grid):
            mass_std = max(0.0, np.polyval(self.std_mass_fit, L))
            ixx_std = max(0.0, np.polyval(self.std_ixx_fit, L))
            iyy_std = max(0.0, np.polyval(self.std_iyy_fit, L))
            izz_std = max(0.0, np.polyval(self.std_izz_fit, L))

            km_kf_avg = abs(np.polyval(self.avg_km_kf_fit, L))
            km_kf_std = max(0.0, np.polyval(self.std_km_kf_fit, L))
            while km_kf_avg - km_kf_std < 5e-4:
                km_kf_std *= 0.9

            volumes[i] = mass_std * ixx_std * iyy_std * izz_std * km_kf_std

        total = volumes.sum()
        if total > 0:
            self.arm_length_grid = L_grid
            self.arm_length_pmf = volumes / total
        else:
            self.arm_length_grid = None
            self.arm_length_pmf = None

    def sample_arm_length(self, rng):
        if self.arm_length_pmf is not None:
            return rng.choice(self.arm_length_grid, p=self.arm_length_pmf)
        return rng.uniform(self.scale_lengths[0], self.scale_lengths[1])
