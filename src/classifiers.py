import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TranportModeDecisionTree:
    """
    Class for keeping and using transport mode decision
    tree classifier.
    """

    def __init__(
        self,
        decision_tree: DecisionTreeClassifier
    ):
        """
        Constructs TranportModeDecisionTree.

        Parameters
        ----------
            decision_tree: DecisionTreeClassifier
                Trained sklearn decision tree for transport mode
                classification.
        """

        self.decision_tree = decision_tree

    def __call__(
        self,
        inputs: np.array
    ) -> int:
        """
        Returns decision tree predicition.

        Parameters
        ----------
            inputs: np.array
                Inputs values array structured equal to
                TransportModeInputs.get_input_vector() result.

        Returns
        -------
            prediction: int
                Predicton of transport mode coded as int with dict
                {
                    'komunikacja samochodowa': 0,
                    'komunikacja zbiorowa': 1,
                    'pieszo': 2,
                    'rower': 3,
                }
        """

        predicition = self.decision_tree.predict([inputs])[0]

        return int(predicition)
