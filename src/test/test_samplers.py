from collections import Counter
from typing import Dict

from ..samplers import RegionSampler, AgeSexSampler


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
