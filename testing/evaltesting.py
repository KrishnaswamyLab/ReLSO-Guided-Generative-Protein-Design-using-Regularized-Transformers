
import unittest
from argparse import ArgumentParser
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


from relso.utils import eval_utils

# test smoothness calcs
class TestEvalUtils(unittest.TestCase):

    def setUp(self):
        self.test_embeddings = np.random.randn(100, 50)
        self.test_targets = np.random.randn(100)
        self.test_preds = np.random.randn(100)


    def test_weighted_smooth(self):
        b_smooth_val, w_smooth_val = eval_utils.get_smoothnes_knn_weighted(self.test_embeddings,
                                                          self.test_targets)
        self.assertIsNotNone(b_smooth_val)
        self.assertTrue(b_smooth_val > 0.0)

        self.assertIsNotNone(w_smooth_val)
        self.assertTrue(w_smooth_val > 0.0)



    def test_gamma_from_sparse(self):

        A = kneighbors_graph(self.test_embeddings,
                             n_neighbors=5, 
                             mode='distance')        
        # get gamma val
        gamma_val = eval_utils.get_gamma_from_sparse(A)


        self.assertIsNotNone(gamma_val)
        self.assertTrue(gamma_val > 0.0)


    def test_percent_error(self):
        perc_error = eval_utils.get_percent_error(self.test_preds, self.test_targets)


        self.assertIsNotNone(perc_error)
        self.assertTrue(perc_error > 0.0)

    def test_pearson_r(self):

        pearson_r = eval_utils.get_pearson_r2(self.test_preds, self.test_targets)

        self.assertIsNotNone(pearson_r)

    def test_spearman_r(self):
        spearman_r = eval_utils.get_spearman_r2(self.test_preds, self.test_targets)

        self.assertIsNotNone(spearman_r)

    
if __name__ == '__main__':
    unittest.main()
