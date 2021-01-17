from typing import Dict

from pytest import fixture

from .data_models import TransportModeInputs


@fixture(scope='session')
def region_prob_dist() -> Dict[str, float]:
    return {
        '1': 0.5,
        '2': 0.3,
        '3': 0.2
    }


@fixture(scope='session')
def demography_dist() -> Dict[str, float]:
    return {
        "0-5": 0.06757193242806758,
        "6-15_K": 0.044112955887044114,
        "6-15_M": 0.044112955887044114,
        "16-19_K": 0.013827486172513828,
        "16-19_M": 0.013827486172513828,
        "20-24_K": 0.019773980226019776,
        "20-24_M": 0.019773980226019776,
        "25-44_K": 0.15951384048615952,
        "25-44_M": 0.15951384048615952,
        "45-60_K": 0.0985886014113986,
        "45-65_M": 0.12049717950282052,
        "61-x_K": 0.1528868871131129,
        "66-x_M": 0.085998874001126
    }


@fixture(scope='session')
def transport_mode_inputs() -> TransportModeInputs:
    return TransportModeInputs(
        age=2,
        pub_trans_comfort=3,
        pub_trans_punctuality=2,
        bicycle_infrastr_comfort=0,
        pedestrian_inconvenience=6,
        household_persons=2,
        household_cars=1,
        household_bicycles=2
    )


@fixture(scope='session')
def pub_trans_comfort_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        },
        "45-65_M": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        }
    }


@fixture(scope='session')
def pub_trans_punctuality_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        },
        "45-65_M": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        }
    }


@fixture(scope='session')
def bicycle_infrastr_comfort_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        },
        "45-65_M": {
            "0": 0.1,
            "1": 0.2,
            "2": 0.2,
            "3": 0.25,
            "4": 0.25
        }
    }


@fixture(scope='session')
def pedestrian_inconvenience_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.1,
            "1": 0.1,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1,
            "5": 0.1,
            "6": 0.1,
            "7": 0.1,
            "8": 0.04,
            "9": 0.04,
            "10": 0.04,
            "11": 0.04,
            "12": 0.04
        },
        "45-65_M": {
            "0": 0.1,
            "1": 0.1,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1,
            "5": 0.1,
            "6": 0.1,
            "7": 0.1,
            "8": 0.04,
            "9": 0.04,
            "10": 0.04,
            "11": 0.04,
            "12": 0.04
        }
    }


@fixture(scope='session')
def household_persons_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "1": 0.2,
            "2": 0.2,
            "3": 0.1,
            "4": 0.1,
            "5": 0.1,
            "6": 0.1,
            "7": 0.1,
            "8": 0.1,
        },
        "45-65_M": {
            "1": 0.2,
            "2": 0.2,
            "3": 0.1,
            "4": 0.1,
            "5": 0.1,
            "6": 0.1,
            "7": 0.1,
            "8": 0.1,
        }
    }


@fixture(scope='session')
def household_cars_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.4,
            "1": 0.3,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1
        },
        "45-65_M": {
            "0": 0.4,
            "1": 0.3,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1
        }
    }


@fixture(scope='session')
def household_bicycles_dist() -> Dict[str, float]:
    return {
        "16-19_K": {
            "0": 0.4,
            "1": 0.3,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1
        },
        "45-65_M": {
            "0": 0.4,
            "1": 0.3,
            "2": 0.1,
            "3": 0.1,
            "4": 0.1
        }
    }


@fixture(scope='session')
def travels_num_dist() -> Dict[str, Dict[str, float]]:
    return {
        "16-19_K": {
            "0": 0.4,
            "2": 0.4,
            "3": 0.1,
            "4": 0.1
        },
        "45-65_M": {
            "0": 0.4,
            "2": 0.4,
            "3": 0.1,
            "4": 0.1
        }
    }


@fixture(scope='session')
def start_hour_dist() -> Dict[str, Dict[str, float]]:
    return {
        'praca': {
            "5": 0.05,
            "6": 0.05,
            "7": 0.15,
            "8": 0.05,
            "9": 0.05,
            "10": 0.05,
            "11": 0.05,
            "12": 0.05,
            "13": 0.05,
            "14": 0.05,
            "15": 0.05,
            "16": 0.05,
            "17": 0.05,
            "18": 0.05,
            "19": 0.05,
            "20": 0.05,
            "21": 0.05,
            "22": 0.05,
            "23": 0.05
        },
        'inne': {
            "5": 0.05,
            "6": 0.05,
            "7": 0.15,
            "8": 0.05,
            "9": 0.05,
            "10": 0.05,
            "11": 0.05,
            "12": 0.05,
            "13": 0.05,
            "14": 0.05,
            "15": 0.05,
            "16": 0.05,
            "17": 0.05,
            "18": 0.05,
            "19": 0.05,
            "20": 0.05,
            "21": 0.05,
            "22": 0.05,
            "23": 0.05
        },
        'dom': {
            "5": 0.05,
            "6": 0.05,
            "7": 0.15,
            "8": 0.05,
            "9": 0.05,
            "10": 0.05,
            "11": 0.05,
            "12": 0.05,
            "13": 0.05,
            "14": 0.05,
            "15": 0.05,
            "16": 0.05,
            "17": 0.05,
            "18": 0.05,
            "19": 0.05,
            "20": 0.05,
            "21": 0.05,
            "22": 0.05,
            "23": 0.05
        }
    }


@fixture(scope='session')
def dest_type_dist() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        "16-19_K": {
            "dom": {
                "dom": 0.1,
                "praca": 0.5,
                "inne": 0.4
            },
            "praca": {
                "dom": 0.5,
                "praca": 0.1,
                "inne": 0.4
            },
            "inne": {
                "dom": 0.4,
                "praca": 0.4,
                "inne": 0.2
            }
        },
        "45-65_M": {
            "dom": {
                "dom": 0.1,
                "praca": 0.5,
                "inne": 0.4
            },
            "praca": {
                "dom": 0.5,
                "praca": 0.1,
                "inne": 0.4
            },
            "inne": {
                "dom": 0.4,
                "praca": 0.4,
                "inne": 0.2
            }
        },
    }


@fixture(scope='session')
def spend_time_dist_params() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {
        "16-19_K": {
            "dom": {
                "loc": 120,
                "scale": 30
            },
            "praca": {
                "loc": 640,
                "scale": 60
            },
            "inne": {
                "loc": 120,
                "scale": 30
            },
        },
        "45-65_M": {
            "dom": {
                "loc": 120,
                "scale": 30
            },
            "praca": {
                "loc": 640,
                "scale": 60
            },
            "inne": {
                "loc": 120,
                "scale": 30
            },
        },
    }
