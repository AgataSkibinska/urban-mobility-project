from typing import Dict, List

# import geopandas as gpd
import numpy as np

from .data_models import ScheduleElement, TransportModeInputs


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

        assert 0.99 < round(sum(prob_dist.values()), 4) <= 1

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

    def __call__(self) -> int:
        """
            Sample value from normal distribution with sampler's params.

            Returns
            -------
                sample: int
                    Sampled integer.
        """

        sample = max(
            int(np.random.normal(self.loc, self.scale)),
            self.min_value
        )

        return sample


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


class TransportModeInputsSampler:
    """
    Sampler class to select agents' input values for transport mode
    classifier.
    """

    def __init__(
        self,
        pub_trans_comfort_dist: Dict[str, Dict[str, float]],
        pub_trans_punctuality_dist: Dict[str, Dict[str, float]],
        bicycle_infrastr_comfort_dist: Dict[str, Dict[str, float]],
        pedestrian_inconvenience_dist: Dict[str, Dict[str, float]],
        household_persons_dist: Dict[str, Dict[str, float]],
        household_cars_dist: Dict[str, Dict[str, float]],
        household_bicycles_dist: Dict[str, Dict[str, float]]
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
        """

        self.pub_trans_comfort_samplers = {}
        for age_sex, input_dist in pub_trans_comfort_dist.items():
            self.pub_trans_comfort_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.pub_trans_punctuality_samplers = {}
        for age_sex, input_dist in pub_trans_punctuality_dist.items():
            self.pub_trans_punctuality_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.bicycle_infrastr_comfort_samplers = {}
        for age_sex, input_dist in bicycle_infrastr_comfort_dist.items():
            self.bicycle_infrastr_comfort_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.pedestrian_inconvenience_samplers = {}
        for age_sex, input_dist in pedestrian_inconvenience_dist.items():
            self.pedestrian_inconvenience_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.household_persons_samplers = {}
        for age_sex, input_dist in household_persons_dist.items():
            self.household_persons_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.household_cars_samplers = {}
        for age_sex, input_dist in household_cars_dist.items():
            self.household_cars_samplers[age_sex] = BaseSampler(
                input_dist
            )

        self.household_bicycles_samplers = {}
        for age_sex, input_dist in household_bicycles_dist.items():
            self.household_bicycles_samplers[age_sex] = BaseSampler(
                input_dist
            )

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
                pub_trans_comfort=int(
                    self.pub_trans_comfort_samplers[
                        age_sex
                    ]()
                ),
                pub_trans_punctuality=int(
                    self.pub_trans_punctuality_samplers[
                        age_sex
                    ]()
                ),
                bicycle_infrastr_comfort=int(
                    self.bicycle_infrastr_comfort_samplers[
                        age_sex
                    ]()
                ),
                pedestrian_inconvenience=int(
                    self.pedestrian_inconvenience_samplers[
                        age_sex
                    ]()
                ),
                household_persons=int(
                    self.household_persons_samplers[
                        age_sex
                    ]()
                ),
                household_cars=int(
                    self.household_cars_samplers[
                        age_sex
                    ]()
                ),
                household_bicycles=int(
                    self.household_bicycles_samplers[
                        age_sex
                    ]()
                )
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
        any_travel_dist: Dict[str, Dict[str, float]],
        travel_chains_dist: Dict[str, Dict[str, float]],
        start_hour_dist: Dict[str, Dict[str, float]],
        other_travels_dist: Dict[str, Dict[str, float]],
        spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
        trip_cancel_prob: Dict[str, float]
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

        self.any_travel_samplers = {}
        for age_sex, dist in any_travel_dist.items():
            self.any_travel_samplers[age_sex] = BaseSampler(
                dist
            )

        self.travel_chains_samplers = {}
        for age_sex, dist in travel_chains_dist.items():
            self.travel_chains_samplers[age_sex] = BaseSampler(
                dist
            )

        self.start_hours_samplers = {}
        for dest_type, dist in start_hour_dist.items():
            self.start_hours_samplers[dest_type] = BaseSampler(
                dist
            )

        self.other_travels_samplers = {}
        for age_sex, dist in other_travels_dist.items():
            self.other_travels_samplers[age_sex] = BaseSampler(
                dist
            )

        self.spend_time_samplers = {}
        for age_sex in spend_time_dist_params.keys():
            self.spend_time_samplers[age_sex] = {}
            for place_type, params in spend_time_dist_params[age_sex].items():
                self.spend_time_samplers[age_sex][
                    place_type
                ] = BaseNormalSampler(
                    loc=params['loc'],
                    scale=params['scale'],
                    min_value=10
                )

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
        
        any_travel = self.any_travel_samplers[
            age_sex
        ]()

        if age_sex != "0-5" and any_travel == '1':

            travel_chain = self.travel_chains_samplers[
                age_sex
            ]().split(',')

            first_destination = travel_chain[0]

            if first_destination == 'inne':
                first_destination_with_other_split = self.other_travels_samplers[age_sex]()
            else:
                first_destination_with_other_split = first_destination
            
            first_start_time = int(
                self.start_hours_samplers[first_destination]()
            ) * 60 + self._sample_minutes()
            first_spend_time = self.spend_time_samplers[age_sex][
                first_destination_with_other_split
            ]()

            if self.trip_cancel_prob[first_destination_with_other_split] <= np.random.random():
                # do not cancel this trip, so add it to schedule
                schedule.append(
                    ScheduleElement(
                        start_time=first_start_time,
                        dest_type=first_destination_with_other_split
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
                next_spend_time = self.spend_time_samplers[age_sex][
                    next_destination_with_other_split
                ]()

                if self.trip_cancel_prob[next_destination_with_other_split] <= np.random.random():
                    # do not cancel this trip, so add it to schedule
                    schedule.append(
                        ScheduleElement(
                            start_time=next_start_time,
                            dest_type=next_destination_with_other_split
                        )
                    )
                prev_start_time = next_start_time
                prev_spend_time = next_spend_time

        return schedule

    def _sample_minutes(
        self
    ) -> int:
        return int(np.random.uniform(0, 60))


# class BuildingsSampler:
#     """
#     Sampler class to select building by type in given region.
#     """

#     def __init__(
#         self,
#         buildings_gdf: gpd.GeoDataFrame,
#         regions_centroids_gdf: gpd.GeoDataFrame
#     ):
#         """
#         Constructs BuildingsSampler.

#         Parameters
#         ----------
#             buildings_gdf: GeoDataFrame
#                 GeoDataFrame of buildings with columns ID, Type, geometry,
#                 Region. epsg=4326
#             regions_centroids_gds: GeoDataFrame
#                 GeoDataFrame of regions with columns NUMBER, geometry
#                 where NUMBER is region id. epsg=4326
#         """

#         buildings_types = {
#             'szkola': ['school'],
#             'praca': ['office'],
#             'inne': [
#                 'commercial', 'hotel', 'retail', 'warehouse', 'church',
#                 'kindergarten', 'service', 'civic', 'kiosk', 'gymnasium',
#                 'sports_hall', 'supermarket', 'allotment_house', 'synagogue',
#                 'cathedral', 'religious', 'hostel', 'concert_hall',
#                 'swimming_pool', 'stadium', 'hospital'
#             ],
#             'uczelnia': ['university', 'college'],
#             'dom': [
#                 'yes', 'semidetached_house', 'residential', 'apartments',
#                 'house', 'dormitory'
#             ]}
#         regions = regions_centroids_gdf['NUMBER'].unique()

#         buildings_gdf['objects'] = buildings_gdf.apply(
#             self._row_to_building,
#             axis=1
#         )

#         self.buildings = {}
#         for region in regions:
#             self.buildings[str(region)] = {}

#             region_sub_frame = buildings_gdf[buildings_gdf['Region'] == region]
#             self.buildings[str(region)][
#                 'all'
#             ] = region_sub_frame['objects'].to_list()

#             if len(self.buildings[str(region)]['all']) == 0:
#                 centroid = regions_centroids_gdf[
#                     regions_centroids_gdf['NUMBER'] == region
#                 ].iloc[0]['geometry']
#                 self.buildings[str(region)]['all'].append(
#                     Building(
#                         x=centroid.coords[0][0],
#                         y=centroid.coords[0][1],
#                         type='inne',
#                         region=str(region),
#                         osm_id='region_centroid'
#                     )
#                 )

#             for building_type in list(buildings_types.keys()):
#                 sub_frame = region_sub_frame[region_sub_frame['Type'].isin(
#                     buildings_types[building_type]
#                 )]
#                 selected_buildings = sub_frame['objects'].to_list()

#                 self.buildings[str(region)][building_type] = selected_buildings

#     def _row_to_building(self, row):
#         return Building(
#             x=row['geometry'].centroid.coords[0][0],
#             y=row['geometry'].centroid.coords[0][1],
#             type=row['Type'],
#             region=str(row['Region']),
#             osm_id=str(row['ID'])
#         )

#     def __call__(
#         self,
#         region: str,
#         building_type: str
#     ) -> Building:
#         """
#         Returns building of given type from given region.

#         Parameters
#         ----------
#             region: str
#                 Building region id.
#             building_type: str
#                 Type of building like "szkola", "dom", "praca", "inne",
#                 "uczelnia".

#         Returns
#         -------
#             building: Building
#                 Sampled building.
#         """

#         if len(self.buildings[region][building_type]) > 0:
#             building = np.random.choice(
#                 self.buildings[region][building_type]
#             )
#         else:
#             building = np.random.choice(
#                 self.buildings[region]['all']
#             )

#         return building


class GravitySampler:
    """
    Sampler class to select destination region based on
    start region and travel destination type.
    """

    def __init__(
        self,
        gravity_dist: Dict[str, Dict[str, Dict[str, float]]]
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

        self.dest_region_samplers = {}
        for dest_type in gravity_dist.keys():
            self.dest_region_samplers[dest_type] = {}
            for start_region, dist in gravity_dist[dest_type].items():
                self.dest_region_samplers[dest_type][
                    start_region
                ] = BaseSampler(
                    dist
                )

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

        dest_region = self.dest_region_samplers[dest_type][start_region]()

        return dest_region


class DriverSampler:
    """
    Sampler class to select driver from those who have chosen to bravel by car.
    """
    def __init__(
        self,
        drivers_dist: Dict[str, Dict[str, float]]
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
        self.drivers_samplers = {}
        for age_sex, input_dist in drivers_dist.items():
            self.drivers_samplers[age_sex] = BaseSampler(
                input_dist
            )

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
