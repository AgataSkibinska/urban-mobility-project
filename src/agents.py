from mesa import Agent

from classifiers import TranportModeDecisionTree
from samplers import RegionSampler, AgeSexSampler, \
    TransportModeInputsSampler, DayScheduleSampler


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
        drivers_dist,  # grouped by age_sex_comb among car travels
    ):
        super().__init__(unique_id, model)

        self.home_region = home_region_sampler()
        self.age_sex = age_sex_sampler()
        self.transport_mode_inputs = transport_mode_inputs_sampler(
            self.age_sex
        )
        self.schedule = day_schedule_sampler(self.age_sex)
        self.transport_mode_clf = transport_mode_clf

        self.drivers_dist = drivers_dist

        self.current_place = self._sample_building(self.home_region, "home")
        self.on_the_way = False
        self.travel_end_time = None # time of simulation when agent finish current travel
        self.travel_destination = None # place where agent finish current travel


    def step(
        self,
        time: int # simulation time in seconds
    ):
        
        if self.on_the_way:

            if self.travel_end_time < time:
                self._finish_travel()
                self._travel()

        else:
            self._travel()


    def _travel(self, time):

        while self._should_start_new_travel():
            
            self._start_new_travel()

            if self.travel_end_time < time:
                self._finish_travel()


    def _finish_travel(self):
        self.on_the_way = False
        self.current_place = self.travel_destination


    def _should_start_new_travel(self, time):

        should_start = False
        if self.day_schedule[0][0] < time and not self.on_the_way:
                should_start = True

        return should_start


    def _start_new_travel(self):

        start_time, destination_type = self.day_schedule.pop(0)[1]

        travel_destination = self._sample_travel_destination(destination_type)

        travel_mode = self.transport_mode_clf.predict([[
            # clf input
        ]])
        travel_time = self._trip_planner(
            self.current_place,
            travel_destination,
            travel_mode
        )

        self.on_the_way = True
        self.travel_end_time = start_time + travel_time
        self.travel_destination = travel_destination


    def _sample_travel_destination(self, destination_type):
        
        current_region = self._place_to_region(self.current_place)
        destination_region = self._gravity(current_region, destination_type)

        destination_place = self._sample_building(destination_region, destination_type)

        return destination_place
        

    def _place_to_region(self, place):
        pass

    def _gravity(self, current_region, destination_type):
        pass
         