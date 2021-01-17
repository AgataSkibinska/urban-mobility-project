from sklearn.tree import DecisionTreeClassifier

from ..classifiers import TranportModeDecisionTree
from ..data_models import TransportModeInputs


def test_transport_mode_decision_tree(
    decision_tree: DecisionTreeClassifier,
    transport_mode_inputs: TransportModeInputs
):

    tranport_mode_tree = TranportModeDecisionTree(
        decision_tree=decision_tree
    )

    prediction = tranport_mode_tree(
        transport_mode_inputs.get_input_vector(150)
    )

    assert type(prediction) == int
    assert prediction in [0, 1, 2, 3]
