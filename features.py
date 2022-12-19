import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class JSBSimFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, 17)

    def forward(self, observations):

        # Unpack
        position = observations[:,:3]
        mach = observations[:,3:4]
        alpha_beta = observations[:,4:6]
        angular_rates = observations[:,6:9]
        phi_theta = observations[:,9:11]
        psi =  observations[:,11:12]
        goal = observations[:,12:]

        # Transform position
        displacement = goal - position
        distance = th.sqrt(th.sum(displacement[:,:2]**2, 1, True))
        dz = displacement[:,2:3]
        altitude = position[:,2:3]
        abs_bearing = th.atan2(displacement[:,1:2], displacement[:,0:1])
        rel_bearing = abs_bearing - psi

        # We normalize distance this way to bound it between 0 and 1
        dist_norm = 1/(1+distance*1e-3)

        # Normalize these by approximate flight ceiling
        dz_norm = dz/15000
        alt_norm = altitude/15000

        # Angles to Sine/Cosine pairs
        cab, sab = th.cos(alpha_beta), th.sin(alpha_beta)
        cpt, spt = th.cos(phi_theta), th.sin(phi_theta)
        cr, sr = th.cos(rel_bearing), th.sin(rel_bearing)

        return th.concat([dist_norm, dz_norm, alt_norm, mach, angular_rates, cab, sab, cpt, spt, cr, sr], 1)