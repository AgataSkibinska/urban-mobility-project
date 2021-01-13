from mesa import Agent

class Person(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # self.start_region = self._sample_region()
        # self.current_region = self.start_region
        # self.on_the_way = self._sample_status()
        # self.end_region = self.start_region
        
        # if self.on_the_way:
        #     self.regions_num_left = self._sample_regions_num() - 1
        # else:
        #     self.regions_num_left = 0

        # self.regions_num = self.regions_num_left

        self.home_region

        self.day_schedule

        self.age

        #

    def _sample_region(self):

        sample = np.random.multinomial(1, list(population_dist.values()))
        sample = np.argmax(sample)
        region = list(population_dist.keys())[sample]

        return region

    def _sample_status(self):

        try:
            sample = np.random.multinomial(1, list(travel_dist[str(self.start_region)].values()))
        except KeyError:
            sample = np.random.multinomial(1, np.zeros(2))
        status = not np.argmax(sample)

        return status

    def _sample_regions_num(self):

        try:
            sample = np.random.multinomial(1, list(regions_num_dist[str(self.start_region)].values()))
        except KeyError:
            sample = np.random.multinomial(1, np.zeros(33))
        regions_num = np.argmax(sample) + 1

        return regions_num

    def move(self):

        # current_region = regions[regions['NUMBER'] == int(self.current_region)].iloc[0]
        # neighbors = regions[~regions.geometry.disjoint(current_region.geometry)]
        # neighbors = neighbors['NUMBER'].to_list()

        # self.current_region = np.random.choice(neighbors)
        # self.regions_num_left = self.regions_num_left - 1

        self.current_region = self._sample_next_region()
        self.regions_num_left = self.regions_num_left - 1

    def _sample_next_region(self):

        sample = np.random.multinomial(1, list(neighbors_dist[str(self.current_region)].values()))
        sample = np.argmax(sample)
        next_region = int(list(neighbors_dist[str(self.current_region)].keys())[sample])

        return next_region

    def step(self):
        
        if self.on_the_way:

            self.move()            

            if self.regions_num_left == 0:
                self.on_the_way = False
                self.end_region = self.current_region