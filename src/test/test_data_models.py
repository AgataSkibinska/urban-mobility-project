import numpy as np

from ..data_models import TransportModeInputs


def test_transport_mode_inputs(
    transport_mode_inputs: TransportModeInputs
):

    distance = 150.0

    model_input = transport_mode_inputs.get_input_vector(
        distance=distance
    )

    assert np.array_equal(
        model_input,
        np.array([
            transport_mode_inputs.pub_trans_comfort,
            transport_mode_inputs.pub_trans_punctuality,
            transport_mode_inputs.bicycle_infrastr_comfort,
            transport_mode_inputs.pedestrian_inconvenience,
            transport_mode_inputs.household_persons,
            transport_mode_inputs.age,
            transport_mode_inputs.household_cars,
            transport_mode_inputs.household_bicycles,
            distance
        ])
    )
