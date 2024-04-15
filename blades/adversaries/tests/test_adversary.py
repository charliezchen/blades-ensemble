import unittest

import ray
import torch

from fedlib.datasets import ToyFLDataset
from fedlib.datasets import DatasetCatalog

from blades.algorithms.fedavg import FedavgConfig
from blades.adversaries import (
    ALIEAdversary,
    IPMAdversary,
    SignFlipAdversary,
    LabelFlipAdversary,
    AdaptiveAdversary,
)


class TestAdversary(unittest.TestCase):
    """Tests for the adversary class."""

    def setUp(self):
        DatasetCatalog.register_custom_dataset("simple", ToyFLDataset)

        self.algorithm = (
            FedavgConfig()
            .data(
                # num_clients=2,
                dataset_config={"custom_dataset": "simple"},
            )
            .training(global_model=torch.nn.Linear(2, 2))
            .adversary(
                num_malicious_clients=1,
                adversary_config={"type": "blades.adversaries.AdaptiveAdversary"},
            )
            .resources(num_remote_workers=1, num_gpus_per_worker=0)
            .build()
        )
        self.adversary = self.algorithm.callbacks[-1]
        self.adversary.setup(self.algorithm)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_algorithm_start(self):
        """Tests the on_algorithm_start method."""
        # self.adversary.setup(self.algorithm)
        for client in self.adversary.clients:
            self.assertTrue(client.is_malicious)

    def test_config(self):
        """Tests the config method."""
        all_advs = [
            ALIEAdversary,
            IPMAdversary,
            SignFlipAdversary,
            LabelFlipAdversary,
            AdaptiveAdversary,
        ]

        for adv_cls in all_advs:
            config = (
                FedavgConfig()
                .resources(num_remote_workers=0, num_gpus_per_worker=0)
                .data(
                    # num_clients=2,
                    dataset_config={
                        "custom_dataset": "simple",
                        # "num_classes": 2,
                        # "custom_dataset_config": {"num_classes": 2},
                    },
                )
                .training(global_model=torch.nn.Linear(2, 2))
            )
            if adv_cls == IPMAdversary:
                adv = (
                    config.adversary(adversary_config={"type": adv_cls, "scale": 0.1})
                    .build()
                    .callbacks[-1]
                )
            else:
                adv = (
                    config.adversary(adversary_config={"type": adv_cls})
                    .build()
                    .callbacks[-1]
                )
            self.assertIsInstance(adv, adv_cls)


if __name__ == "__main__":
    unittest.main()
