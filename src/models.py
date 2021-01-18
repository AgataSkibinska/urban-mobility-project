from typing import Dict

from mesa import Model
from mesa.time import RandomActivation
from sklearn.tree import DecisionTreeClassifier

from .agents import Person
from .classifiers import TranportModeDecisionTree
from .samplers import (AgeSexSampler, DayScheduleSampler, DriverSampler,
                      GravitySampler, RegionSampler,
                      TransportModeInputsSampler)


class TrafficModel(Model):
    """A model with some number of agents."""
    def __init__(
        self,
        N: int,
        population_dist: Dict[str, float],
        demography_dist: Dict[str, float],
        pub_trans_comfort_dist: Dict[str, Dict[str, float]],
        pub_trans_punctuality_dist: Dict[str, Dict[str, float]],
        bicycle_infrastr_comfort_dist: Dict[str, Dict[str, float]],
        pedestrian_inconvenience_dist: Dict[str, Dict[str, float]],
        household_persons_dist: Dict[str, Dict[str, float]],
        household_cars_dist: Dict[str, Dict[str, float]],
        household_bicycles_dist: Dict[str, Dict[str, float]],
        travels_num_dist: Dict[str, Dict[str, float]],
        start_hour_dist: Dict[str, Dict[str, float]],
        dest_type_dist: Dict[str, Dict[str, Dict[str, float]]],
        spend_time_dist_params: Dict[str, Dict[str, Dict[str, int]]],
        decision_tree: DecisionTreeClassifier,
        gravity_dist: Dict[str, Dict[str, Dict[str, float]]],
        drivers_dist: Dict[str, Dict[str, float]],
        interregional_distances: Dict[str, Dict[str, float]],
        start_time: int = 4 * 60,
        step_time: int = 60
    ):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.running = True
        self.start_time = start_time
        self.step_time = step_time

        # All Agent subclasses init
        self.home_region_sampler = RegionSampler(
            prob_dist=population_dist
        )
        self.age_sex_sampler = AgeSexSampler(
            prob_dist=demography_dist
        )
        self.mode_inputs_sampler = TransportModeInputsSampler(
            pub_trans_comfort_dist=pub_trans_comfort_dist,
            pub_trans_punctuality_dist=pub_trans_punctuality_dist,
            bicycle_infrastr_comfort_dist=bicycle_infrastr_comfort_dist,
            pedestrian_inconvenience_dist=pedestrian_inconvenience_dist,
            household_persons_dist=household_persons_dist,
            household_cars_dist=household_cars_dist,
            household_bicycles_dist=household_bicycles_dist
        )
        self.day_schedule_sampler = DayScheduleSampler(
            travels_num_dist=travels_num_dist,
            start_hour_dist=start_hour_dist,
            dest_type_dist=dest_type_dist,
            spend_time_dist_params=spend_time_dist_params
        )
        self.transport_mode_clf = TranportModeDecisionTree(
            decision_tree=decision_tree
        )
        self.gravity_sampler = GravitySampler(
            gravity_dist=gravity_dist
        )
        self.driver_sampler = DriverSampler(
            drivers_dist=drivers_dist
        )
        self.interregional_distances = interregional_distances

        # Create agents
        for i in range(self.num_agents):
            a = Person(
                unique_id=i,
                model=self,
                home_region_sampler=self.home_region_sampler,
                age_sex_sampler=self.age_sex_sampler,
                transport_mode_inputs_sampler=self.mode_inputs_sampler,
                day_schedule_sampler=self.day_schedule_sampler,
                transport_mode_clf=self.transport_mode_clf,
                gravity_sampler=self.gravity_sampler,
                driver_sampler=self.driver_sampler,
                interregional_distances=self.interregional_distances,
                start_time=self.start_time,
                step_time=self.step_time
            )
            self.schedule.add(a)

        # self.datacollector = DataCollector(
        #     agent_reporters={
        #         "regions_num": "regions_num",
        #         "start_region": "start_region",
        #         "end_region": "end_region"
        #     }
        # )

    def step(self):
        # self.datacollector.collect(self)
        self.schedule.step()
