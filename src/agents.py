from typing import Dict

from mesa import Agent

from .classifiers import TranportModeDecisionTree
from .samplers import (AgeSexSampler, DayScheduleSampler, DriverSampler,
                       GravitySampler, RegionSampler,
                       TransportModeInputsSampler)


class Person(Agent):

    def __init__(
        self,
        unique_id,
        model,
        home_region_sampler: RegionSampler,
        age_sex_sampler: AgeSexSampler,
        transport_mode_inputs_sampler: TransportModeInputsSampler,
        day_schedule_sampler: DayScheduleSampler,
        transport_mode_clf: TranportModeDecisionTree,
        gravity_sampler: GravitySampler,
        driver_sampler: DriverSampler,
        interregional_distances: Dict[str, Dict[str, float]],
        start_time: int,
        step_time: int
    ):
        super().__init__(unique_id, model)

        self.home_region = home_region_sampler()
        self.age_sex = age_sex_sampler()
        self.schedule = day_schedule_sampler(self.age_sex)
        self.transport_mode_inputs = transport_mode_inputs_sampler(
            self.age_sex
        )

        self.transport_mode_clf = transport_mode_clf
        self.gravity_sampler = gravity_sampler
        self.driver_sampler = driver_sampler
        self.interregional_distances = interregional_distances

        self.step_time = step_time

        self.current_region = self.home_region
        self.current_place_type = 'dom'
        self.current_time = start_time

        # Additional fields for agent data collector
        self.agent_id = unique_id
        #  home_region already exists
        #  age_sex already exists
        self.pub_trans_comfort = self.transport_mode_inputs.pub_trans_comfort
        self.pub_trans_punctuality = self.transport_mode_inputs.pub_trans_punctuality
        self.bicycle_infrastr_comfort = self.transport_mode_inputs.bicycle_infrastr_comfort
        self.pedestrian_inconvenience = self.transport_mode_inputs.pedestrian_inconvenience
        self.household_persons = self.transport_mode_inputs.household_persons
        self.household_cars = self.transport_mode_inputs.household_cars
        self.household_bicycles = self.transport_mode_inputs.household_bicycles
        self.travels_num = len(self.schedule)

        # Additional fields for travels data collector
        #  self.agent_id already exists
        self.start_region = []
        self.start_place_type = []
        self.dest_region = []
        self.dest_place_type = []
        self.travel_start_time = []
        self.dest_activity_dur_time = []
        self.transport_mode = []
        self.is_driver = []

    def step(self):
        while self._should_start_new_travel(
            time=self.current_time
        ):
            self._start_new_travel()

        self.current_time += self.step_time

    def _should_start_new_travel(
        self,
        time: int
    ):
        return len(self.schedule) > 0 and self.schedule[0].travel_start_time <= time

    def _start_new_travel(self):
        schedule_element = self.schedule.pop(0)

        if schedule_element.dest_activity_type == 'dom':
            dest_region = self.home_region
        else:
            dest_region = self.gravity_sampler(
                start_region=self.current_region,
                dest_type=schedule_element.dest_activity_type
            )

        distance = self.interregional_distances[
            self.current_region
        ][
            dest_region
        ]

        travel_mode = self.transport_mode_clf(
            self.transport_mode_inputs.get_input_vector(
                distance=distance
            )
        )

        is_driver = None
        if travel_mode == 0:
            is_driver = self.driver_sampler(
                age_sex=self.age_sex
            )

        # update fields for travels data collector
        self.start_region.append(self.current_region)
        self.start_place_type.append(self.current_place_type)
        self.dest_region.append(dest_region)
        self.dest_place_type.append(schedule_element.dest_activity_type)
        self.travel_start_time.append(schedule_element.travel_start_time)
        self.dest_activity_dur_time.append(schedule_element.dest_activity_dur_time)
        self.transport_mode.append(travel_mode)
        self.is_driver.append(is_driver)

        self.current_region = dest_region
        self.current_place_type = schedule_element.dest_activity_type
