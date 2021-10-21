from collections import Counter
from typing import Dict

import numpy as np

from ..samplers import (AgeSexSampler, BaseNormalSampler, BaseSampler,
                        DayScheduleSampler, DriverSampler,
                        GravitySampler, RegionSampler,
                        TransportModeInputsSampler)


def test_base_sampler_1():
    dist = {
        "A": 1.,
        "B": 0
    }

    sampler = BaseSampler(dist)

    samples = [sampler() for i in range(100000)]

    assert samples == ["A" for i in range(100000)]


def test_base_sampler_2():
    dist = {
        "A": 0.5,
        "B": 0.3,
        "C": 0.2
    }

    sampler = BaseSampler(dist)

    samples = [sampler() for i in range(100000)]

    assert samples.count("A") > samples.count("B") > samples.count("C")


def test_base_normal_sampler():
    loc = 100
    scale = 10
    min_value = 10

    norm_sampler = BaseNormalSampler(
        loc=loc,
        scale=scale,
        min_value=min_value
    )

    samples = np.array([norm_sampler() for i in range(100000)])

    assert 99 < samples.mean() < 101
    assert 9 < samples.std() < 11
    assert min(samples) >= min_value


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


def test_day_schedule_sampler_1(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "0": 1
        },
        "45-65_M": {
            "0": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob
    )

    schedule = day_schedule_sampler("16-19_K")

    assert schedule == []


def test_day_schedule_sampler_2(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "2": 1
        },
        "45-65_M": {
            "2": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) == 2
    # assert schedule[0].dest_activity_type in ['dom', 'praca', 'inne']
    assert schedule[0].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[1].dest_activity_type == 'dom'
    assert schedule[0].travel_start_time < schedule[1].travel_start_time
    assert 0 <= schedule[0].travel_start_time
    assert 0 <= schedule[1].travel_start_time


def test_day_schedule_sampler_3(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "3": 1
        },
        "45-65_M": {
            "3": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) == 3
    # assert schedule[0].dest_activity_type in ['dom', 'praca', 'inne']
    assert schedule[0].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    # assert schedule[1].dest_activity_type in ['dom', 'praca', 'inne']
    assert schedule[1].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[2].dest_activity_type == 'dom'
    assert schedule[0].travel_start_time < schedule[1].travel_start_time
    assert schedule[1].travel_start_time < schedule[2].travel_start_time
    assert 0 <= schedule[0].travel_start_time
    assert 0 <= schedule[1].travel_start_time


def test_day_schedule_sampler_4(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "5": 1
        },
        "45-65_M": {
            "5": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) == 5
    assert schedule[0].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[1].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[2].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[3].dest_activity_type in [
        'dom', 'praca', 'culture_and_entertainment',
        'gastronomy', 'grocery_shopping'
    ]
    assert schedule[4].dest_activity_type == 'dom'
    assert schedule[0].travel_start_time < schedule[1].travel_start_time
    assert schedule[1].travel_start_time < schedule[2].travel_start_time
    assert schedule[2].travel_start_time < schedule[3].travel_start_time
    assert schedule[3].travel_start_time < schedule[4].travel_start_time
    assert 0 <= schedule[0].travel_start_time
    assert 0 <= schedule[1].travel_start_time
    assert 0 <= schedule[1].travel_start_time
    assert 0 <= schedule[2].travel_start_time
    assert 0 <= schedule[3].travel_start_time


def test_day_schedule_sampler_5(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob_2: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "5": 1
        },
        "45-65_M": {
            "5": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob_2
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) == 0


def test_day_schedule_sampler_6(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob_3: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "5": 1
        },
        "45-65_M": {
            "5": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob_3
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) <= 5
    for schedule_element in schedule:
        assert schedule_element.dest_activity_type in [
            'dom', 'gastronomy', 'grocery_shopping'
        ]


def test_day_schedule_sampler_7(
    travels_num_dist: Dict[str, Dict[str, float]],
    start_hour_dist: Dict[str, Dict[str, float]],
    dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
    other_travels_dist: Dict[str, Dict[str, float]],
    spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
    trip_cancel_prob: Dict[str, float]
):
    travels_num_dist = {
        "16-19_K": {
            "5": 1
        },
        "45-65_M": {
            "5": 1
        }
    }

    day_schedule_sampler = DayScheduleSampler(
        travels_num_dist=travels_num_dist,
        start_hour_dist=start_hour_dist,
        dest_type_dist=dest_type_dist,
        other_travels_dist=other_travels_dist,
        spend_time_dist_params=spend_time_dist_params,
        trip_cancel_prob=trip_cancel_prob
    )

    schedule = day_schedule_sampler("16-19_K")

    assert len(schedule) == 5
    for i in range(len(schedule)-1):
        start_time = schedule[i].travel_start_time
        end_time = schedule[i+1].travel_start_time
        dur_time = end_time - start_time
        assert schedule[i].dest_activity_dur_time == dur_time

# def test_buildings_sampler(
#     buildings_gdf: gpd.GeoDataFrame,
#     regions_centroids_gdf: gpd.GeoDataFrame
# ):
#     buildings_sampler = BuildingsSampler(
#         buildings_gdf=buildings_gdf,
#         regions_centroids_gdf=regions_centroids_gdf
#     )

#     building_1 = buildings_sampler(
#         region='1',
#         building_type='dom'
#     )

#     # 209 is region with no buildings
#     building_2 = buildings_sampler(
#         region='209',
#         building_type='dom'
#     )

#     assert type(building_1) == Building
#     assert 17.03027 <= building_1.x <= 17.03859
#     assert 51.10658 <= building_1.y <= 51.11240
#     assert type(building_2) == Building
#     assert round(building_2.x, 6) == round(16.89475215576043, 6)
#     assert round(building_2.y, 6) == round(51.18178842847963551, 6)


def test_gravity_sampler(
    gravity_dist: Dict[str, Dict[str, Dict[str, float]]]
):
    gravity_sampler = GravitySampler(
        gravity_dist=gravity_dist
    )

    dest_region_1 = gravity_sampler(
        start_region="1",
        dest_type="praca"
    )
    dest_region_2 = gravity_sampler(
        start_region="2",
        dest_type="culture_and_entertainment"
    )

    assert type(dest_region_1) == str
    assert dest_region_1 in ["1", "2", "3"]
    assert type(dest_region_2) == str
    assert dest_region_2 == "1"


def test_driver_sampler(
    drivers_dist: Dict[str, Dict[str, float]]
):
    driver_sampler = DriverSampler(
        drivers_dist=drivers_dist
    )

    driver_1 = driver_sampler(
        age_sex='16-19_K'
    )

    driver_2 = driver_sampler(
        age_sex='45-65_M'
    )

    assert type(driver_1) == str
    assert driver_1 in ["0", "1"]
    assert type(driver_2) == str
    assert driver_2 == "0"
