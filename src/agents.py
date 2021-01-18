from typing import Dict

from mesa import Agent

from classifiers import TranportModeDecisionTree
from samplers import (AgeSexSampler, DayScheduleSampler, DriverSampler,
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
        interregional_distances: Dict[str, Dict[str, float]]
    ):
        super().__init__(unique_id, model)

        self.home_region = home_region_sampler()
        self.age_sex = age_sex_sampler()
        self.transport_mode_inputs = transport_mode_inputs_sampler(
            self.age_sex
        )
        self.schedule = day_schedule_sampler(self.age_sex)

        self.transport_mode_clf = transport_mode_clf
        self.gravity_sampler = gravity_sampler
        self.driver_sampler = driver_sampler
        self.interregional_distances = interregional_distances

        self.current_region = self.home_region

    def step(
        self,
        time: int
    ):
        while self._should_start_new_travel(
            time=time
        ):
            self._start_new_travel()

    def _should_start_new_travel(
        self,
        time: int
    ):

        return len(self.schedule) > 0 and self.schedule[0].start_time <= time

    def _start_new_travel(self):
        schedule_element = self.day_schedule.pop(0)

        dest_region = self.gravity_sampler(
            start_region=self.current_region,
            dest_type=schedule_element.dest_type
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

        self.current_region = dest_region
