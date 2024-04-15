import unittest
from unittest.mock import MagicMock

import ray
import torch
from fedlib.constants import CLIENT_UPDATE, CLIENT_ID

from blades.adversaries import AdaptiveAdversary


class TestAdaptiveAdversary(unittest.TestCase):
    def setUp(self):
        self.clients = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        for client in self.clients:
            client.is_malicious = False
        self.global_config = {"lr": 0.1}
        self.algorithm = MagicMock()
        self.scale = 3
        self.device = "cpu"
        # self.device = "cuda:0"
        self.algorithm.local_results = [
            {
                CLIENT_ID: 0,
                CLIENT_UPDATE: torch.tensor([-4, -7, 1, 3], device=self.device),
            },
            {
                CLIENT_ID: 1,
                CLIENT_UPDATE: torch.tensor([-1, -4, 2, -1], device=self.device),
            },
            {
                CLIENT_ID: 2,
                CLIENT_UPDATE: torch.tensor([0.1, -2, 3, 3], device=self.device),
            },
            {
                CLIENT_ID: 3,
                CLIENT_UPDATE: torch.tensor([3, -1.0, 4, 3], device=self.device),
            },
        ]
        self.algorithm.client_manager.get_client_by_id.side_effect = (
            lambda x: self.clients[x]
        )
        self.adversary = AdaptiveAdversary()
        self.adversary.setup(self.algorithm)
        self.adversary.on_trainer_init(self.algorithm)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_on_local_round_end(self):
        output = self.adversary.on_local_round_end(self.algorithm)
        lower_bound = torch.tensor(
            [3, -1, 1 / self.scale, -1 * self.scale],
            device=self.device,
        )
        upper_bound = torch.tensor(
            [self.scale * 3, -1 / self.scale, 1, -1],
            device=self.device,
        )
        self.assertTrue(
            torch.all(lower_bound <= output) and torch.all(output <= upper_bound)
        )


if __name__ == "__main__":
    unittest.main()
