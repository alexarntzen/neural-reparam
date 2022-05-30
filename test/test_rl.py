import unittest

import spinup
from neural_reparam.reparam_env import RealReparamEnv

import experiments.curves as c1


class TestEnv(unittest.TestCase):
    def test_gym_reparm_env(self):
        # Load data
        for N in [32]:
            spinup.ddpg_pytorch(
                lambda: RealReparamEnv(r_func=c1.r, q_func=c1.q, size=N),
                steps_per_epoch=2000,
                epochs=50,
                replay_size=int(1e5),
                gamma=0.99,
                polyak=0.995,
                pi_lr=1e-3,
                q_lr=1e-3,
                batch_size=100,
                start_steps=200,
                update_after=1000,
                update_every=10,
                act_noise=0.2,
                num_test_episodes=10,
                max_ep_len=N**2,
                save_freq=1,
            )
            # ddpg_pytorch(lambda: RealReparamEnv(r_func=c1.r, q_func=c1.q, size=N))


if __name__ == "__main__":
    unittest.main()
