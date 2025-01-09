import unittest

import numpy as np
from scipy import sparse

from autosklearn.pipeline.components.data_preprocessing.NoPreprocessing import NoPreprocessing


class NoPreprocessingTest(unittest.TestCase):
    def test_preprocessing_dtype_transform(self):
        # Dense
        X = np.random.rand(3, 2)
        Y = np.random.randint(0, 2, (3,))
        no_preprocessing = NoPreprocessing()
        no_preprocessing.fit(X, Y)
        X_transformed = no_preprocessing.transform(X)
        self.assertIsInstance(X_transformed, np.ndarray)
        np.testing.assert_array_equal(X_transformed, X)

        # Sparse
        X = sparse.csr_matrix(X)
        Y = np.random.randint(0, 2, (3,))
        no_preprocessing = NoPreprocessing()
        no_preprocessing.fit(X, Y)
        X_transformed = no_preprocessing.transform(X)
        self.assertIsInstance(X_transformed, sparse.csr_matrix)
        np.testing.assert_array_equal(X_transformed.toarray(), X.toarray())

    def test_preprocessing_dtype_transform_no_fit(self):
        X = np.random.rand(3, 2)
        no_preprocessing = NoPreprocessing()
        with self.assertRaises(NotImplementedError):
            no_preprocessing.transform(X)

    def test_preprocessing_properties(self):
        props = NoPreprocessing.get_properties()
        self.assertEqual(props['shortname'], 'NoPreprocessing')
        self.assertTrue(props['handles_regression'])
        self.assertTrue(props['handles_classification'])
        self.assertTrue(props['handles_multiclass'])
        self.assertTrue(props['handles_multilabel'])
        self.assertTrue(props['handles_multioutput'])
        self.assertTrue(props['is_deterministic'])

    def test_hyperparameter_search_space(self):
        cs = NoPreprocessing.get_hyperparameter_search_space()
        self.assertEqual(len(cs.get_hyperparameters()), 0)