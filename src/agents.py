from mesa import Agent

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
        transport_mode_clf,
        drivers_dist,  # grouped by age_sex_comb among car travels
    ):
        super().__init__(unique_id, model)

        self.home_region = home_region_sampler()
        self.age_sex = age_sex_sampler()
        self.transport_mode_inputs = transport_mode_inputs_sampler(
            self.age_sex
        )
        self.schedule = day_schedule_sampler(self.age_sex)

        # Possible output:
        #   'komunikacja samochodowa': 0,
        #   'komunikacja zbiorowa': 1,
        #   'pieszo': 2,
        #   'rower': 3,
        self.transport_mode_clf = transport_mode_clf
        self.drivers_dist = drivers_dist

        """
        Atrybuty wejściowe do drzewa decyzyjnego:
        (ich rozkłady de względu na wiek i płeć)

        *1'Wygoda jazdy komunikacją',
        *1'Punktualność komunikacji ',
        *1'Ocena systemu rowerowego',
        *2'Piesze niedogodności',
        'Liczba osób',  # liczba osób w gospodarstwie domowym (ogółem)
        'Przedział wiekowy',
        'Samochód', # liczba samochodów w gospodarstwie domowym
        'Rower', # liczba rowerów w gospodarstwie domowym
        'Liczba przebytych rejonów'

        *1
            'bardzo źle': 0,
            'raczej źle': 1,
            'ani dobrze ani źle': 2,
            'nie korzystam z komunikacji zbiorowej': 2,
            'raczej dobrze': 3,
            'bardzo dobrze': 4
        *2
            0-12 (więcej -> więcej uciązliwości) 
        """

        self.home_region = self._sample_region()
        self.age_sex_comb = self._sample_agent_demography()
        self.age = self.age_sex_comb.split('_')[0]
        # self.age, self.sex = self.age_sex_comb.split('_')

        self.pub_trans_comfort = self._sample_with_dist(self.pub_trans_comfort_dist)
        self.pub_trans_punctuality = self._sample_with_dist(self.pub_trans_punctuality_dist)
        self.bicycle_infrastr_comfort = self._sample_with_dist(self.bicycle_infrastr_comfort_dist)
        self.pedestrian_inconvenience = self._sample_with_dist(self.pedestrian_inconvenience_dist)
        self.household_persons = self._sample_with_dist(self.household_persons_dist)
        self.household_cars = self._sample_with_dist(self.household_cars_dist)
        self.household_bicycles = self._sample_with_dist(self.household_bicycles_dist)

        # TODO losowanie planu dnia
        # można wykorzystać: travels_num_dist; destination_dist
        # [(czas1, cel1), (czas2, cel2), ...] = foo   # dokładny czas
        self.day_schedule = self._sample_day_schedule()

        self.current_place = self._sample_building(self.home_region, "home")
        self.on_the_way = False
        self.travel_end_time = None # time of simulation when agent finish current travel
        self.travel_destination = None # place where agent finish current travel 


    def _sample_region(self):
        sample = np.random.multinomial(1, list(self.population_dist.values()))
        sample = np.argmax(sample)
        region = list(self.population_dist.keys())[sample]

        return region
    

    def _sample_agent_demography(self):
        sample = np.random.multinomial(1, list(self.demography_dist.values()))
        sample = np.argmax(sample)
        agent_sex_comb = list(self.demography_distr.keys())[sample]

        return agent_sex_comb

    
    def _sample_with_dist(self, dist):
        pass

    
    def _sample_day_schedule(self):
        pass

    
    def _sample_building(self, region, tag):
        pass


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
         