import os
import requests
import json
import datetime


class OpenTripPlannerRequest:

    def __init__(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        host: str,
        port: str,
        transport_mode: int,
        is_driver: bool,
        date: datetime.datetime,

    ):
        """Parameters
        ----------
        x_start, x_end - source, destination longitude
        y_start, y_end - source, destination latitude
        host, port - otp instance params
        transport_mode - transport mode returned by decision tree
        is_driver - if transport mode is 0 (CAR)
        date - trip start datetime
        """
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.host = host
        self.port = port
        self.transport_mode_dict = {
                     0: 'CAR',
                     1: 'TRANSIT',
                     2: 'WALK',
                     3: 'BICYCLE'
                }
        self.transport_mode = self.transport_mode_dict[transport_mode]
        self.is_driver = is_driver
        self.date = date.strftime("%m-%d-%Y")  #mm-dd-yyyy
        self.time = date.strftime("%I:%M %p")

    def create_request_string(self) -> str:
        """Returns
        -------
        Request url"""
        return f"http://{self.host}:{self.port}/otp/routers/default/plan/"

    def get_response(self) -> json:
        """Returns
        -------
        Whole response from open trip planner"""
        url = self.create_request_string()
        payload = {'fromPlace': f'{self.y_start},{self.x_start}',
                   'toPlace': f'{self.y_end},{self.x_end}',
                   'mode': self.transport_mode,
                   'date': self.date,
                   'time': self.time}
        r = requests.get(url, params=payload)
        return r.json()

    @staticmethod
    def get_trip_duration(json_file):
        """Returns
        -------
        Selected trip scenario duration"""
        duration = json_file['plan']['itineraries'][0]['duration']
        return duration/60

    def parse_and_save_result(self, json_file):
        """Saves selected trip scenario as json file"""
        parsed_json_file = json_file['plan']['itineraries'][0]
        path = self.create_output_file_path()
        with open(path, "w") as write_file:
            json.dump(parsed_json_file, write_file)

    def create_output_file_path(self) -> str:
        """Returns
        -------
        File path based on init params"""
        if self.transport_mode == 'CAR':
            return f"simulation_data/{self.transport_mode}_{str(self.is_driver)}" \
                   f"/trip_{datetime.datetime.now().timestamp()}.json "
        else:
            return f"simulation_data/{self.transport_mode}/trip_{datetime.datetime.now().timestamp()}.json"

    @staticmethod
    def create_dirs():
        """Creates obligatory dirs"""
        os.makedirs("simulation_data/CAR_false")
        os.makedirs("simulation_data/CAR_true")
        os.makedirs("simulation_data/TRANSIT")
        os.makedirs("simulation_data/BICYCLE")
        os.makedirs("simulation_data/WALK")
