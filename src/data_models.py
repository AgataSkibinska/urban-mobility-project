from dataclasses import dataclass

import numpy as np


@dataclass
class TransportModeInputs:
    """
    Class for keeping transport mode classifier inputs.

    Attributes
    ----------
        age : int
            Respondent age mapped with dict:
            {
                '6-15 (dzieci)': 0,
                '16-19 (młodzież)': 1,
                '20-24 (wiek studencki)': 2,
                '25-44 (młodsi pracownicy)': 3,
                '45-60 (starsi pracownicy kobiety)': 4,
                '45-65 (starsi pracownicy mężczyźni)': 4,
                '61 i więcej (emeryci kobiety)': 5,
                '66 i więcej (emeryci mężczyźni)': 5
            }.
        pub_trans_comfort : int
            Answer for question "Jak Pan/Pani ocenia wygodę jazdy pojazdami
            komunikacji zbiorowej?" mapped with dict:
            {
                'bardzo źle': 0,
                'raczej źle': 1,
                'ani dobrze ani źle': 2,
                'nie korzystam z komunikacji zbiorowej': 2,
                'raczej dobrze': 3,
                'bardzo dobrze': 4
            }.
        pub_trans_punctuality: int
            Answer for question "Jak Pan/Pani ocenia punktualność komunikacji
            zbiorowej we Wrocławiu?" mapped with dict:
            {
                'bardzo źle': 0,
                'raczej źle': 1,
                'ani dobrze ani źle': 2,
                'nie korzystam z komunikacji zbiorowej': 2,
                'raczej dobrze': 3,
                'bardzo dobrze': 4
            }.
        bicycle_infrastr_comfort: int
            Answer for question "Jak ocenia Pan/Pani efekty dotychczasowych
            działań związanych z rozbudową systemu rowerowego we Wrocławiu
            (dróg i parkingów rowerowych)?" mapped with dict:
            {
                'bardzo źle': 0,
                'raczej źle': 1,
                'ani dobrze ani źle': 2,
                'nie korzystam z komunikacji zbiorowej': 2,
                'nie korzystam z systemu dróg i parkingów rowerowych': 2,
                'raczej dobrze': 3,
                'bardzo dobrze': 4
            }.
        pedestrian_inconvenience: int
            Summarized value representing answers for questions about
            pedestrian inconveniences:
                - "Niekorzystne ustawienia sygnalizacji świetlnej",
                - "Brak chodników i konieczność poruszania się
                    jezdnią/poboczem/wydeptaną ścieżką",
                - "Zły stan nawierzchni chodników",
                - "Zastawianie chodników przez parkujące samochody",
                - "Niebezpieczne zachowania kierowców",
                - "Zagrożenie ze strony rowerzystów poruszających się
                    chodnikami",
                - "Zbyt wysokie krawężniki",
                - "Brak bieżącego utrzymania czystości/odśnieżania",
                - "Niewystarczająca liczba przejść dla pieszych",
                - "Brak miejsc wypoczynku na trasie dojścia (np. ławki,
                    zieleń)",
                - "Uciążliwy ruch kołowy",
                - "Niewłaściwe oświetlenie ciągów pieszych".
            All ansers were mapped with dict:
            {
                'Nie': 0,
                'nie': 0,
                'Tak': 1,
                'tak': 1
            }
            and then summed up.
        household_persons : int
            Number of persons on household from questionnaires column
            "Liczba osób w gospodarstwie domowym [ogółem]".
        household_cars : int
            Summed up number of cars on household from questionnaires column:
                - "Samochód prywatny, zarejestrowany na osobę
                    z gosp. domowego",
                - "Samochód prywatny, nie zarejestrowany na osobę
                    z gosp. domowego [użyczone]"
                - "Samochód służbowy".
        household_bicycles : int
            Number of bicycles on household from column "Rower".
    """

    age: int
    pub_trans_comfort: int
    pub_trans_punctuality: int
    bicycle_infrastr_comfort: int
    pedestrian_inconvenience: int
    household_persons: int
    household_cars: int
    household_bicycles: int

    def get_input_vector(
        self,
        distance: float
    ) -> np.array:
        """
        Returns array of collected attributes and given distance between
        regions - transport mode classifier input for some travel.

        Parameters
        ----------
            distance : float
                Travel distance.

        Returns
        -------
            input_array: np.array
                Input array for transport mode classifier.
        """

        return np.array([
            self.pub_trans_comfort,
            self.pub_trans_punctuality,
            self.bicycle_infrastr_comfort,
            self.pedestrian_inconvenience,
            self.household_persons,
            self.age,
            self.household_cars,
            self.household_bicycles,
            distance
        ])


@dataclass
class ScheduleElement:
    """
    Class for keeping element of day schedule list.

    Attributes
    ----------
        travel_start_time: int
            Travel start time in minutes of day. Take a note that for
            second and each next travel the start time field is a time 
            of destination activity start without estimation of travel
            time between activities.
        dest_activity_type: str
            Travel destination type like "szkola", "uczelnia", "dom",
            "praca", "inne"
        dest_activity_dur_time: int
            Duration time (minutes) of activity that is this travel
            destination. Default value is 0.
    """

    travel_start_time: int
    dest_activity_type: str
    dest_activity_dur_time: int = 0


# @dataclass
# class Building:
#     """
#     Class for keeping information about building.

#     Attributes
#     ----------
#         x: float
#             X coord of building.
#         y: float
#             Y coord of building.
#         type: str
#             Building type like "szkola", "dom", "praca", "inne", "uczelnia".
#         region: str
#             Building region id.
#         osm_id: str
#             Building id from OSM.
#     """

#     x: float
#     y: float
#     type: str
#     region: str
#     osm_id: str
