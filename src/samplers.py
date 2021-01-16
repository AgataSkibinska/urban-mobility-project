from typing import Dict

import numpy as np


class BaseSampler:
    """
    Sampler class to select some object for given distribution.
    """

    def __init__(
        self,
        prob_dist: Dict[str, float]
    ):
        """
        Constructs BaseSampler with given probability distribution.

        Parameters
        ----------
            prob_dist : dict
                Dictionary {object_id: str : probability: float} contains
                probabilities of selecting elements.
        """

        self.prob_dist = prob_dist

    def __call__(
        self
    ) -> str:
        """
        Sample object.

        Returns
        -------
            object_id : str
        """

        object_ids = list(self.prob_dist.keys())
        probs = list(self.prob_dist.values())

        sample = np.random.multinomial(1, probs)
        sample = np.argmax(sample)
        object_id = object_ids[sample]

        return object_id


class RegionSampler(BaseSampler):
    """
    Sampler class to select region for given distribution.
    """

    def __init__(
        self,
        prob_dist: Dict[str, float]
    ):
        BaseSampler.__init__(
            self,
            prob_dist
        )


class AgeSexSampler(BaseSampler):
    """
    Sampler class to select age and sex combination for given distribution.
    Combinations are strings like: "0-5", "6-15_K", "6-15_M" ...
    ... "45-60_K", "45-65_M", "61-x_K", "66-x_M".
    """

    def __init__(
        self,
        prob_dist: Dict[str, float]
    ):
        BaseSampler.__init__(
            self,
            prob_dist
        )
