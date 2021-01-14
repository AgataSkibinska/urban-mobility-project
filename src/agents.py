from mesa import Agent

class Person(Agent):

    def __init__(
        self,
        unique_id,
        model,
        transport_mode_clf,
        population_dist,
        demography_dist,
        travels_num_dist,                       # grouped by age_sex_comb
        destination_dist,                       # grouped by age_sex_comb
        pub_trans_comfort_dist,                 # grouped by age_sex_comb
        pub_trans_punctuality_dist,             # grouped by age_sex_comb
        bicycle_infrastr_comfort_dist,          # grouped by age_sex_comb
        pedestrian_inconvenience_dist,          # grouped by age_sex_comb
        household_persons_dist,                 # grouped by age_sex_comb
        household_cars_dist,                    # grouped by age_sex_comb
        household_bicycles_dist,                # grouped by age_sex_comb
        drivers_dist,                           # grouped by age_sex_comb among car travels
    ):
        super().__init__(unique_id, model)

        # Possible output:
        #   'komunikacja samochodowa': 0,
        #   'komunikacja zbiorowa': 1,
        #   'pieszo': 2,
        #   'rower': 3,
        self.transport_mode_clf = transport_mode_clf

        self.population_dist = population_dist
        self.demography_dist = demography_dist
        self.travels_num_dist = travels_num_dist
        self.destination_dist = destination_dist
        self.pub_trans_comfort_dist = pub_trans_comfort_dist
        self.pub_trans_punctuality_dist = pub_trans_punctuality_dist
        self.bicycle_infrastr_comfort_dist = bicycle_infrastr_comfort_dist
        self.pedestrian_inconvenience_dist = pedestrian_inconvenience_dist
        self.household_persons_dist = household_persons_dist
        self.household_cars_dist = household_cars_dist
        self.household_bicycles_dist = household_bicycles_dist
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


    # def _sample_status(self):

    #     try:
    #         sample = np.random.multinomial(1, list(travel_dist[str(self.start_region)].values()))
    #     except KeyError:
    #         sample = np.random.multinomial(1, np.zeros(2))
    #     status = not np.argmax(sample)

    #     return status

    # def _sample_regions_num(self):

    #     try:
    #         sample = np.random.multinomial(1, list(regions_num_dist[str(self.start_region)].values()))
    #     except KeyError:
    #         sample = np.random.multinomial(1, np.zeros(33))
    #     regions_num = np.argmax(sample) + 1

    #     return regions_num

    # def move(self):

    #     # current_region = regions[regions['NUMBER'] == int(self.current_region)].iloc[0]
    #     # neighbors = regions[~regions.geometry.disjoint(current_region.geometry)]
    #     # neighbors = neighbors['NUMBER'].to_list()

    #     # self.current_region = np.random.choice(neighbors)
    #     # self.regions_num_left = self.regions_num_left - 1

    #     self.current_region = self._sample_next_region()
    #     self.regions_num_left = self.regions_num_left - 1

    # def _sample_next_region(self):

    #     sample = np.random.multinomial(1, list(neighbors_dist[str(self.current_region)].values()))
    #     sample = np.argmax(sample)
    #     next_region = int(list(neighbors_dist[str(self.current_region)].keys())[sample])

    #     return next_region

    def step(self):
        
        if self.on_the_way:

            self.move()            

            if self.regions_num_left == 0:
                self.on_the_way = False
                self.end_region = self.current_region