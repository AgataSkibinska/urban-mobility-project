from typing import Any, Dict, List, Tuple

# import geopandas as gpd
import numpy as np
from numpy.random import default_rng, MT19937


from .data_models import ScheduleElement, TransportModeInputs


GLOBAL_SEED = 2137
GLOBAL_RNG = default_rng(MT19937(GLOBAL_SEED))
RNG_BUFFER = 32


DistributionFloatTuple = Tuple[List[str], List[float]]

class BaseSampler:
    """
    Sampler class to select some object for given distribution.
    """

    def __init__(
        self,
        prob_dist: Tuple[List[Any], List[float]],
        num_samples: int = RNG_BUFFER,
    ):
        """
        Constructs BaseSampler with given probability distribution.

        Parameters
        ----------
            prob_dist : dict
                Dictionary {object_id: T : probability: float} contains
                probabilities of selecting elements.
        """

        self.object_ids_samples = None
        self.object_ids = prob_dist[0]
        self.probs = prob_dist[1]
        self.rng = GLOBAL_RNG
        self.counter = 0
        self.num_samples = num_samples

        # assert 0.99 < np.around(np.sum(self.probs), decimals=2) <= 1

    
    def __call__(
        self
    ) -> Any:
        """
        Sample object.

        Returns
        -------
            object_id : T
        """

        if self.counter % self.num_samples == 0:
            self.counter = self.counter % self.num_samples
            self.object_ids_samples = self.rng.choice(
                self.object_ids,
                size=self.num_samples,
                p=self.probs
            )
        
        value = self.object_ids_samples[self.counter]
        self.counter = self.counter + 1

        return value


class BaseNormalSampler:
    """
    Class to sample integer from normal distribution.
    """

    def __init__(
        self,
        loc: int,
        scale: int,
        min_value: int
    ):
        """
        Constructs NormalSampler with given params.

        Parameters
        ----------
            loc: int
                Mean value - normaln distribution parameter.
            scale: int
                Std - normaln distribution parameter.
            min_value: int
                Minimal output value. If sampled value is less than
                min_value then min_value is returned.
        """

        assert scale >= 0

        self.loc = loc
        self.scale = scale
        self.min_value = min_value
        self.samples = None
        self.rng = GLOBAL_RNG
        self.counter = 0

    def __call__(self) -> int:
        """
            Sample value from normal distribution with sampler's params.

            Returns
            -------
                sample: int
                    Sampled integer.
        """

        if self.counter % RNG_BUFFER == 0:
            self.samples = np.maximum(
                np.floor(
                    self.rng.normal(self.loc, self.scale, RNG_BUFFER)
                ),
                self.min_value
            )

        value = self.samples[self.counter]
        self.counter = (self.counter + 1) % RNG_BUFFER

        return value


class RegionSampler(BaseSampler):
    """
    Sampler class to select region for given distribution.
    """

    def __init__(
        self,
        prob_dist: DistributionFloatTuple,
        num_samples: int = RNG_BUFFER
    ):
        BaseSampler.__init__(
            self,
            prob_dist,
            num_samples
        )


class AgeSexSampler(BaseSampler):
    """
    Sampler class to select age and sex combination for given distribution.
    Combinations are strings like: "0-5", "6-15_K", "6-15_M" ...
    ... "45-60_K", "45-65_M", "61-x_K", "66-x_M".
    """

    def __init__(
        self,
        prob_dist: DistributionFloatTuple,
        num_samples: int = RNG_BUFFER
    ):
        BaseSampler.__init__(
            self,
            prob_dist,
            num_samples
        )


class TransportModeInputsSampler:
    """
    Sampler class to select agents' input values for transport mode
    classifier.
    """

    
    def __init__(
        self,
        pub_trans_comfort_dist: List[Tuple[str, DistributionFloatTuple]],
        pub_trans_punctuality_dist: List[Tuple[str, DistributionFloatTuple]],
        bicycle_infrastr_comfort_dist: List[Tuple[str, DistributionFloatTuple]],
        pedestrian_inconvenience_dist: List[Tuple[str, DistributionFloatTuple]],
        household_persons_dist: List[Tuple[str, DistributionFloatTuple]],
        household_cars_dist: List[Tuple[str, DistributionFloatTuple]],
        household_bicycles_dist: List[Tuple[str, DistributionFloatTuple]],
        num_samples: int = RNG_BUFFER
    ):
        """
        Constructs TransportModeInputsSampler with given probability
        distributions.

        Parameters
        ----------
            pub_trans_comfort_dist: dict
                Dictionary
                {age_sex: str : {
                    pub_trans_comfort: str : probability: float}
                }
                contains probabilities for pub_trans_comfort input. Sampled
                pub_trans_comfort will be converted to int.
            pub_trans_punctuality_dist: dict
                Dictionary
                {age_sex: str : {
                    pub_trans_punctuality: str : probability: float}
                }
                contains probabilities for pub_trans_punctuality input.
                Sampled pub_trans_punctuality will be converted to int.
            bicycle_infrastr_comfort_dist: dict
                Dictionary
                {age_sex: str : {
                    bicycle_infrastr_comfort: str : probability: float}
                }
                contains probabilities for bicycle_infrastr_comfort input.
                Sampled bicycle_infrastr_comfort will be converted to int.
            pedestrian_inconvenience_dist: dict
                Dictionary
                {age_sex: str : {
                    pedestrian_inconvenience: str : probability: float}
                }
                contains probabilities for pedestrian_inconvenience input.
                Sampled pedestrian_inconvenience will be converted to int.
            household_persons_dist: dict
                Dictionary
                {age_sex: str : {
                    household_persons: str : probability: float}
                }
                contains probabilities for household_persons input.
                Sampled household_persons will be converted to int.
            household_cars_dist: dict
                Dictionary
                {age_sex: str : {
                    household_cars: str : probability: float}
                }
                contains probabilities for household_cars input.
                Sampled household_cars will be converted to int.
            household_bicycles_dist: dict
                Dictionary
                {age_sex: str : {
                    household_bicycles: str : probability: float}
                }
                contains probabilities for household_bicycles input.
                Sampled household_bicycles will be converted to int.
            num_samples: int
                Number of actors to somehow adjust BaseSampler
        """

        approx_num_samples = num_samples // 12

        self.pub_trans_comfort_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in pub_trans_comfort_dist
        }

        self.pub_trans_punctuality_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in pub_trans_punctuality_dist
        }

        self.bicycle_infrastr_comfort_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in bicycle_infrastr_comfort_dist
        }

        self.pedestrian_inconvenience_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in pedestrian_inconvenience_dist
        }

        self.household_persons_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in household_persons_dist
        }

        self.household_cars_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in household_cars_dist
        }

        self.household_bicycles_samplers = {
            age_sex: BaseSampler((input_dist[0].astype(np.int), input_dist[1]), approx_num_samples)
            for age_sex, input_dist in household_bicycles_dist
        }

    
    def __call__(
        self,
        age_sex: str
    ) -> TransportModeInputs:
        """
        Samples and returns TransportModeInputs

        Parameters
        ----------
            age_sex: str
                Age and sex combination string like "0-5", "16-19_K"...
                Age from this string will be mapped to int 0-5 value.

        Returns
        -------
            input_values: TransportModeInputs
        """

        age_mapping = {
            "0-5": 0,
            "6-15_K": 0,
            "6-15_M": 0,
            "16-19_K": 1,
            "16-19_M": 1,
            "20-24_K": 2,
            "20-24_M": 2,
            "25-44_K": 3,
            "25-44_M": 3,
            "45-60_K": 4,
            "45-65_M": 4,
            "61-x_K": 5,
            "66-x_M": 5
        }

        if age_sex != "0-5":

            input_values = TransportModeInputs(
                age=age_mapping[age_sex],
                pub_trans_comfort= self.pub_trans_comfort_samplers[age_sex](),
                pub_trans_punctuality=self.pub_trans_punctuality_samplers[age_sex](),
                bicycle_infrastr_comfort=self.bicycle_infrastr_comfort_samplers[age_sex](),
                pedestrian_inconvenience=self.pedestrian_inconvenience_samplers[age_sex](),
                household_persons=self.household_persons_samplers[age_sex](),
                household_cars=self.household_cars_samplers[age_sex](),
                household_bicycles=self.household_bicycles_samplers[age_sex]()
            )

        else:

            input_values = TransportModeInputs(
                age=age_mapping[age_sex],
                pub_trans_comfort=None,
                pub_trans_punctuality=None,
                bicycle_infrastr_comfort=None,
                pedestrian_inconvenience=None,
                household_persons=None,
                household_cars=None,
                household_bicycles=None
            )

        return input_values


class DayScheduleSampler:
    """
    Sampler class to prepare plan of travels for all day.
    """

    
    def __init__(
        self,
        any_travel_dist: List[Tuple[str, DistributionFloatTuple]],
        travel_chains_dist: List[Tuple[str, DistributionFloatTuple]],
        start_hour_dist: List[Tuple[str, DistributionFloatTuple]],
        other_travels_dist: List[Tuple[str, DistributionFloatTuple]],
        spend_time_dist_params: List[Tuple[str,  Dict[str, int]]],
        trip_cancel_prob: List[Tuple[str, float]]
    ):
        """
        Constructs DayScheduleSampler with given probability
        distributions.

        Parameters
        ----------
            any_travel_dist: dict
                Dictionary
                {age_sex: str : {
                    any_travel_occured: int : probability: float}
                }
                contains probabilities of whether any travel has taken place
                (1 means that at least one trip must take place; 0 - no travel).
            travel_chains_dist: dict
                Dictionary
                {age_sex: str : {
                    travel_chain: str : probability: float}
                }
                contains probabilities for specific chains of destinations,
                ordered according to their sequence of execution.
            start_hour_dist: dict
                Dictionary
                {dest_type: str : {
                    hour: str : probability: float}
                }
                contains probabilities for start hours for travel with
                specific destination type. Sampled hour will be converted
                to int.
            other_travels_dist: dict
                Dictionary
                {age_sex: str : {
                    other_place_type: str : probability: float}
                }}
                contains probabilities of performing an activity from
                the available subcategories for the category "inne".
            spend_time_dist_params: dict
                Dictionary
                {age_sex: str : {
                    place_type: str : {
                        "loc" : mean_minutes: int,
                        "scale" : std_minutes: int
                    }
                }}
            trip_cancel_prob: dict
                Dictionary
                {place_type: str : probability: float}}
                contains probabilities of cancellation of activities
                associated with a specific destination place type.
        """

        self.any_travel_samplers = {
            age_sex: BaseSampler(dist) for age_sex, dist in any_travel_dist
        }

        self.travel_chains_samplers = {
            age_sex: BaseSampler(dist) for age_sex, dist in travel_chains_dist
        }

        self.start_hours_samplers = {
            dest_type: BaseSampler((dist[0].astype(np.int), dist[1]))
            for dest_type, dist in start_hour_dist
        }

        self.other_travels_samplers = {
            age_sex: BaseSampler(dist) for age_sex, dist in other_travels_dist
        }

        self.spend_time_samplers = {
            key: BaseNormalSampler(
                    loc=params['loc'],
                    scale=params['scale'],
                    min_value=10
            ) for key, params in spend_time_dist_params
        }

        self.trip_cancel_prob = trip_cancel_prob

    
    def __call__(
        self,
        age_sex: str
    ) -> List[ScheduleElement]:
        """
            Sample day schedule list sorted by travels start time.

            Parameters
            ----------
                age_sex: str
                    Age and sex comination string.

            Returns
            -------
                schedule: list
                    Day travels schedule - list of ScheduleElement.
        """

        schedule = []

        if age_sex != "0-5":
            any_travel = self.any_travel_samplers[
                age_sex
            ]()

            if any_travel == '1':
                travel_chain = self.travel_chains_samplers[
                    age_sex
                ]().split(',')

                cancel_states = GLOBAL_RNG.random(len(travel_chain)).tolist()

                first_destination = travel_chain[0]

                if first_destination == 'inne':
                    first_destination_with_other_split = self.other_travels_samplers[age_sex]()
                else:
                    first_destination_with_other_split = first_destination

                first_start_time = self.start_hours_samplers[first_destination]() * 60 + self._sample_minutes()
                first_spend_time = self.spend_time_samplers[age_sex + first_destination_with_other_split]()

                if self.trip_cancel_prob[first_destination_with_other_split] <= cancel_states.pop():
                    # do not cancel this trip, so add it to schedule
                    schedule.append(
                        ScheduleElement(
                            travel_start_time=first_start_time,
                            dest_activity_type=first_destination_with_other_split,
                            dest_activity_dur_time=first_spend_time
                        )
                    )

                prev_start_time = first_start_time
                prev_spend_time = first_spend_time

                for next_destination in travel_chain[1:]:  # will work fine (min travels in chain = 2)
                    if next_destination == 'inne':
                        next_destination_with_other_split = self.other_travels_samplers[age_sex]()
                    else:
                        next_destination_with_other_split = next_destination

                    next_start_time = prev_start_time + prev_spend_time
                    next_spend_time = self.spend_time_samplers[age_sex + next_destination_with_other_split]()

                    if self.trip_cancel_prob[next_destination_with_other_split] <= cancel_states.pop():
                        # do not cancel this trip, so add it to schedule
                        schedule.append(
                            ScheduleElement(
                                travel_start_time=next_start_time,
                                dest_activity_type=next_destination_with_other_split,
                                dest_activity_dur_time=next_spend_time
                            )
                        )
                    prev_start_time = next_start_time
                    prev_spend_time = next_spend_time

        return schedule

    def _sample_minutes(
        self
    ) -> int:
        return GLOBAL_RNG.integers(0, 60)


class GravitySampler:
    """
    Sampler class to select destination region based on
    start region and travel destination type.
    """

    
    def __init__(
        self,
        gravity_dist: List[Tuple[str, DistributionFloatTuple]]
    ):
        """
        Constructs GravitySampler with given distribution.

        Parameters
        ----------
            gravity_dist: dict
                Dictionary
                {dest_type: str : {
                    start_region_id: str : {
                        dest_region_id: str : prob: float
                    }
                }}
                contains probability for destination region for travel
                with given start region and destination type (like 'szkola'...)
        """

        self.dest_region_samplers = {
            key: BaseSampler(dist, 10)
            for key, dist in gravity_dist
        }


    def __call__(
        self,
        start_region: str,
        dest_type: str
    ) -> str:
        """
        Returns destination region id for given start region id
        and destination type.

        Parameters
        ----------
            start_region: str
                Start region id.
            dest_type: str
                Travel destination type like "szkola", "dom", "praca", "inne",
                "uczelnia".

        Returns
        -------
            dest_region: str
                Sampled destination region id.
        """

        dest_region = self.dest_region_samplers[dest_type + start_region]()
        # print(self.dest_region_samplers[dest_type + start_region].counter)

        return dest_region


class DriverSampler:
    """
    Sampler class to select driver from those who have chosen to bravel by car.
    """
    def __init__(
        self,
        drivers_dist: List[Tuple[str, DistributionFloatTuple]]
    ):
        """
        Constructs DriverSampler with given probability distribution.
        Parameters
        ----------
            drivers_dist: dict
                Dictionary
                {age_sex: str : {
                    is_driver: str : probability: float}
                }
                contains probabilities for drivers_dist input. Sampled
                drivers_dist will be converted to int.
        """
        self.drivers_samplers = { 
            age_sex: BaseSampler(input_dist) for age_sex, input_dist in drivers_dist
        }

    def __call__(
        self,
        age_sex: str
    ) -> str:
        """
        Samples and returns DriverInputs

        Parameters
        ----------
            age_sex: str
                Age and sex combination string like "0-5", "16-19_K"...

        Returns
        -------
            is_driver: str
                Sampled passenger ('0') or driver ('1')
        """

        is_driver = self.drivers_samplers[age_sex]()

        return is_driver
