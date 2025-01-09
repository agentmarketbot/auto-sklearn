from typing import Optional, Dict, Any, Union, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace
import numpy as np

from autosklearn.pipeline.base import DATASET_PROPERTIES_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SPARSE, UNSIGNED_DATA, INPUT


class NoPreprocessing(AutoSklearnPreprocessingAlgorithm):
    def __init__(
        self,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """A component that does no preprocessing, passing the data through unchanged."""
        self.random_state = random_state

    def fit(self, X: DENSE, y: Optional[DENSE] = None) -> "NoPreprocessing":
        """Fit the NoPreprocessing component.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples,), optional
            Targets for supervised learning

        Returns
        -------
        self : NoPreprocessing
            This estimator
        """
        self.fitted_ = True
        return self

    def transform(self, X: DENSE) -> DENSE:
        """Transform the data by doing nothing.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Data to transform

        Returns
        -------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Transformed data (identical to input)
        """
        if self.fitted_ is False:
            raise NotImplementedError()
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            'shortname': 'NoPreprocessing',
            'name': 'No Preprocessing',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'handles_multioutput': True,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (INPUT,)
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        """Return the configuration space for this component.
        
        Returns
        -------
        cs : ConfigurationSpace
            The configuration space describing all hyperparameters of this component.
        """
        cs = ConfigurationSpace()
        return cs