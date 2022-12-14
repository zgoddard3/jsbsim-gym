import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class JSBSimFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, scale=1e-4):
        super().__init__(observation_space, 17)

    def forward(self, observations):

        # Unpack
        position = observations[:3]
        mach, alpha, beta = observations[3:6]
        angular_rates = observations[6:9]
        phi, theta, psi = observations[9:12]
        goal = observations[12:]

        # Transform position
        displacement = goal - position
        distance = th.sqrt(th.sum(displacement[:2]**2))
        dz = displacement[2]
        altitude = position[2]
        abs_bearing = th.atan2(displacement[1], displacement[0])
        rel_bearing = abs_bearing - psi

        # We normalize distance this way to bound it between 0 and 1
        # Note this distance is still in lat/long radians so we scale it up a bit
        dist_norm = 1/(1+1000*distance)

        # Normalize these by approximate flight ceiling
        dz_norm = dz/15000
        alt_norm = altitude/15000

        # Angles to Sine/Cosine pairs
        ca, sa = th.cos(alpha), th.sin(alpha)
        cb, sb = th.cos(beta), th.sin(beta)
        cp, sp = th.cos(phi), th.sin(phi)
        ct, st = th.cos(theta), th.sin(theta)
        cr, sr = th.cos(rel_bearing), th.sin(rel_bearing)

        return th.concat([dist_norm, dz_norm, alt_norm, mach, angular_rates, ca, sa, cb, sb, cp, sp, ct, st, cr, sr], 1)