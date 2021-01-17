from collections import Counter
from typing import Dict

from ..samplers import RegionSampler, AgeSexSampler, TransportModeInputsSampler


def test_region_sampler(
    region_prob_dist: Dict[str, float]
):
    region_sampler = RegionSampler(region_prob_dist)
    single_sample = region_sampler()
    multi_sample = [region_sampler() for i in range(100000)]

    sorted_dist = dict(
        sorted(region_prob_dist.items(), key=lambda item: item[1])
    )
    sorted_samples_counter = dict(
        sorted(Counter(multi_sample).items(), key=lambda item: item[1])
    )

    assert type(single_sample) == str
    assert single_sample in region_prob_dist.keys()
    assert sorted_dist.keys() == sorted_samples_counter.keys()


def test_age_sex_sampler(
    demography_dist: Dict[str, float]
):
    age_sex_sampler = AgeSexSampler(demography_dist)
    single_sample = age_sex_sampler()

    assert type(single_sample) == str
    assert single_sample in demography_dist.keys()


def test_transport_mode_inputs_sampler(
    pub_trans_comfort_dist: Dict[str, Dict[str, float]],
    pub_trans_punctuality_dist: Dict[str, Dict[str, float]],
    bicycle_infrastr_comfort_dist: Dict[str, Dict[str, float]],
    pedestrian_inconvenience_dist: Dict[str, Dict[str, float]],
    household_persons_dist: Dict[str, Dict[str, float]],
    household_cars_dist: Dict[str, Dict[str, float]],
    household_bicycles_dist: Dict[str, Dict[str, float]]
):
    transport_mode_inputs_sampler = TransportModeInputsSampler(
        pub_trans_comfort_dist=pub_trans_comfort_dist,
        pub_trans_punctuality_dist=pub_trans_punctuality_dist,
        bicycle_infrastr_comfort_dist=bicycle_infrastr_comfort_dist,
        pedestrian_inconvenience_dist=pedestrian_inconvenience_dist,
        household_persons_dist=household_persons_dist,
        household_cars_dist=household_cars_dist,
        household_bicycles_dist=household_bicycles_dist
    )

    inputs_1 = transport_mode_inputs_sampler("16-19_K")
    inputs_2 = transport_mode_inputs_sampler("45-65_M")

    assert type(inputs_1.age) == int
    assert inputs_1.age == 1

    assert type(inputs_1.pub_trans_comfort) == int
    assert 0 <= inputs_1.pub_trans_comfort <= 4

    assert type(inputs_1.pub_trans_punctuality) == int
    assert 0 <= inputs_1.pub_trans_punctuality <= 4

    assert type(inputs_1.bicycle_infrastr_comfort) == int
    assert 0 <= inputs_1.bicycle_infrastr_comfort <= 4

    assert type(inputs_1.pedestrian_inconvenience) == int
    assert 0 <= inputs_1.pedestrian_inconvenience <= 12

    assert type(inputs_1.household_persons) == int
    assert 0 <= inputs_1.household_persons <= 8

    assert type(inputs_1.household_cars) == int
    assert 0 <= inputs_1.household_cars <= 4

    assert type(inputs_1.household_bicycles) == int
    assert 0 <= inputs_1.household_bicycles <= 4

    assert type(inputs_2.age) == int
    assert inputs_2.age == 4

    assert type(inputs_2.pub_trans_comfort) == int
    assert 0 <= inputs_2.pub_trans_comfort <= 4

    assert type(inputs_2.pub_trans_punctuality) == int
    assert 0 <= inputs_2.pub_trans_punctuality <= 4

    assert type(inputs_2.bicycle_infrastr_comfort) == int
    assert 0 <= inputs_2.bicycle_infrastr_comfort <= 4

    assert type(inputs_2.pedestrian_inconvenience) == int
    assert 0 <= inputs_2.pedestrian_inconvenience <= 12

    assert type(inputs_2.household_persons) == int
    assert 0 <= inputs_2.household_persons <= 8

    assert type(inputs_2.household_cars) == int
    assert 0 <= inputs_2.household_cars <= 4

    assert type(inputs_1.household_bicycles) == int
    assert 0 <= inputs_1.household_bicycles <= 4
