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
