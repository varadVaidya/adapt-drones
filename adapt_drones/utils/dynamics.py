# file that holds the dynamics class of the drone.
# this class is only to be used along with the option argument of
# the reset function of the environment, to inject arbitrary dynamics
# irrespective of the scale variant nature of the drone.

from dataclasses import dataclass
from adapt_drones.cfgs.config import *


@dataclass
class CustomDynamics:
    """
    This class is only suppoesed to be used with the argument
    of the reset function of the environment, to inject arbitrary
    dynamics irrespective of the scale variant nature of the drone.
    """

    arm_length: float
    mass: float
    ixx: float
    iyy: float
    izz: float
    km_kf: float


class ScaledDynamics:
    """
    Support class used so that the scale variant dynamics can be
    used directly in the MPC evaluation. This is similar in spirit
    to Dynamics class used in the comparison experiments.
    """

    def __init__(self, seed, arm_length, cfg: Config, do_random=True):
        self.arm_length = arm_length
        self.rng: np.random.Generator = seed
        self.do_random = do_random
        self.cfg = cfg

    def length_scale(self):
        return self.arm_length

    def mass_scale(self):
        _mass_avg = np.polyval(self.cfg.scale.avg_mass_fit, self.arm_length)
        _mass_std = np.polyval(self.cfg.scale.std_mass_fit, self.arm_length)
        _mass_std = 0.0 if _mass_std < 0.0 else _mass_std

        _mass_std = 0.0 if not self.do_random else _mass_std

        return self.rng.uniform(_mass_avg - _mass_std, _mass_avg + _mass_std)

    def ixx_yy_scale(self):
        _ixx_avg = np.polyval(self.cfg.scale.avg_ixx_fit, self.arm_length)
        _ixx_std = np.polyval(self.cfg.scale.std_ixx_fit, self.arm_length)
        _ixx_std = 0.0 if _ixx_std < 0.0 else _ixx_std

        _ixx_std = 0.0 if not self.do_random else _ixx_std

        return self.rng.uniform(_ixx_avg - _ixx_std, _ixx_avg + _ixx_std)

    def izz_scale(self):
        _izz_avg = np.polyval(self.cfg.scale.avg_izz_fit, self.arm_length)
        _izz_std = np.polyval(self.cfg.scale.std_izz_fit, self.arm_length)
        _izz_std = 0.0 if _izz_std < 0.0 else _izz_std

        _izz_std = 0.0 if not self.do_random else _izz_std

        return self.rng.uniform(_izz_avg - _izz_std, _izz_avg + _izz_std)

    def torque_to_thrust(self):
        _km_kf_avg = np.abs(np.polyval(self.cfg.scale.avg_km_kf_fit, self.arm_length))
        _km_kf_std = np.polyval(self.cfg.scale.std_km_kf_fit, self.arm_length)
        _km_kf_std = 0.0 if _km_kf_std < 0.0 else _km_kf_std

        while _km_kf_avg - _km_kf_std < 0.0:
            _km_kf_std *= 0.9

        _km_kf_std = 0.0 if not self.do_random else _km_kf_std

        return self.rng.uniform(_km_kf_avg - _km_kf_std, _km_kf_avg + _km_kf_std)
