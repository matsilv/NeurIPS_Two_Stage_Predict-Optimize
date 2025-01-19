"""
    Check if our definition of post-hoc regret for KP01 (in the DFL repo) aligns with the one defined in this repository.
"""

import os.path
import unittest

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from utils import correction_single_obj, StochasticWeightsKnapsackProblem


class MyTestCase(unittest.TestCase):

    _purchase_fee = 0.2
    _compensation_fee = 0.21
    _dataset_filepath = os.path.join('data', 'synthetic')
    _features_filepath = os.path.join(_dataset_filepath, 'features_kp_50.csv')
    _values_filepath = os.path.join(_dataset_filepath, 'values_kp_50.csv')
    _target_filepath = os.path.join(_dataset_filepath, 'targets_kp_50.csv')
    _solutions_filepath = os.path.join(_dataset_filepath, 'solutions_50_kp.csv')
    _capacity_filepath = os.path.join(_dataset_filepath, 'capacity_kp_50.npy')

    @classmethod
    def setUpClass(cls):
        """
        We expected the synthetic data to be placed in the previously defined folder.
        """
        error_msg = f'{cls._dataset_filepath} must exist'
        assert os.path.exists(cls._dataset_filepath), error_msg
        assert os.path.exists(cls._features_filepath)
        assert os.path.exists(cls._values_filepath)
        assert os.path.exists(cls._target_filepath)
        assert os.path.exists(cls._solutions_filepath)
        assert os.path.exists(cls._capacity_filepath)

    def setUp(self):
        """
        Load data before running each test.
        """
        self._features = pd.read_csv(self._features_filepath, index_col=0).values
        self._values = pd.read_csv(self._values_filepath, index_col=0).values
        self._values = np.squeeze(self._values)
        self._weights = pd.read_csv(self._target_filepath, index_col=0).values
        self._solutions = pd.read_csv(self._solutions_filepath, index_col=0).values
        self._capacity = np.load(self._capacity_filepath)

    def test_01_method_works(self):
        """
        Simply check if the method has some semantic errors.

        """
        dataset_idx = 0
        item_num = self._solutions.shape[1]

        values_as_dict = {_idx: self._values[_idx] for _idx in range(item_num)}
        weights_as_dict = {_idx: self._weights[dataset_idx][_idx] for _idx in range(item_num)}

        res = (
            correction_single_obj(values_as_dict,
                                  values_as_dict,
                                  self._capacity,
                                  weights_as_dict,
                                  weights_as_dict,
                                  purchase_fee=self._purchase_fee,
                                  compensation_fee=self._compensation_fee,
                                  item_num=item_num)
        )

        self.assertIsInstance(res.obj, float)
        self.assertIsInstance(res.first_stage_sol, list)
        self.assertEqual(len(res.first_stage_sol), item_num)
        self.assertIsInstance(res.second_stage_sol, list)
        self.assertEqual(len(res.second_stage_sol), item_num)

    def test_02_same_optimal_solutions(self):
        """
        Check if the true optimal solutions are compute the same way.
        """
        item_num = self._solutions.shape[1]
        num_examples = self._solutions.shape[0]

        values_as_dict = {_idx: self._values[_idx] for _idx in range(item_num)}
        weights_as_dict = [{
            _idx: self._weights[_dataset_idx, _idx] for _idx in range(item_num)
        } for _dataset_idx in range(num_examples)]

        for _weights, _expected_sol in tqdm(zip(weights_as_dict, self._solutions),
                                            desc='Checking optimal solutions',
                                            total=len(self._solutions)):
            opt_res = (
                correction_single_obj(values_as_dict,
                                      values_as_dict,
                                      self._capacity,
                                      _weights,
                                      _weights,
                                      purchase_fee=self._purchase_fee,
                                      compensation_fee=self._compensation_fee,
                                      item_num=item_num)
            )

            self.assertTrue((_expected_sol == opt_res.first_stage_sol).all())

    def test_03_same_post_hoc_regret(self):
        """
        Check if the post-hoc regret (and first stage and second stage solutions) are computed the same way.
        """
        item_num = self._solutions.shape[1]
        num_examples = self._solutions.shape[0]

        # Fake predictions.
        pred_weights = np.random.normal(loc=self._weights, scale=0.1)

        # How we define the problem in our repo.
        opt_prob = (
            StochasticWeightsKnapsackProblem(dim=item_num,
                                             purchase_fee=self._purchase_fee,
                                             compensation_fee=self._compensation_fee)
        )
        opt_prob_params = {
            'values': torch.as_tensor(self._values),
            'capacity': torch.as_tensor(self._capacity)
        }

        # This method implementation requires dictionaries for weights and values. See the 'correction_single_obj'
        # method definition.
        values_as_dict = {_idx: self._values[_idx] for _idx in range(item_num)}
        true_weights_as_dict = [{
            _idx: self._weights[_dataset_idx, _idx] for _idx in range(item_num)
        } for _dataset_idx in range(num_examples)]
        pred_weights_as_dict = [{
            _idx: pred_weights[_dataset_idx, _idx] for _idx in range(item_num)
        } for _dataset_idx in range(num_examples)]

        for _true_wgt, _pred_wgt in tqdm(zip(true_weights_as_dict, pred_weights_as_dict),
                                            desc='Post-hoc regret',
                                            total=len(self._solutions)):
            computed_res = (
                correction_single_obj(values_as_dict,
                                      values_as_dict,
                                      self._capacity,
                                      _true_wgt,
                                      _pred_wgt,
                                      purchase_fee=self._purchase_fee,
                                      compensation_fee=self._compensation_fee,
                                      item_num=item_num)
            )

            # Our implementation of the problem requires torch.Tensor for the weights and solutions. See
            # 'StochasticWeightsKnapsackProblem' methods for more details.
            weights_as_array = np.asarray(list(_pred_wgt.values()))
            true_weights_as_tensor = torch.as_tensor(list(_true_wgt.values()))

            expected_fst_stage_sol, _ = (
                opt_prob.solve(y=weights_as_array,
                               opt_prob_params=opt_prob_params)
            )
            expected_fst_stage_sol = torch.as_tensor(expected_fst_stage_sol)

            expected_res = (
                opt_prob.get_objective_values(y=true_weights_as_tensor,
                                              sols=expected_fst_stage_sol,
                                              opt_prob_params=opt_prob_params)
            )

            # There might be some approximation errors.
            self.assertAlmostEqual(expected_res.obj, computed_res.obj, places=7)
            self.assertEqual(expected_res.first_stage_sol, computed_res.first_stage_sol)
            self.assertEqual(expected_res.second_stage_sol, computed_res.second_stage_sol)


if __name__ == '__main__':
    unittest.main()
