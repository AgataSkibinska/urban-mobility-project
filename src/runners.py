import ujson as json
import os
import pickle
from multiprocessing import Pool
import numpy as np

from src.models import TrafficModel
from src.utils import explode

def run(
    in_dir_path: str = '../experiments/input_data/base_distributions',
    out_dir_path: str = '../experiments/results/base_distributions',
    num_agents: int = 635701,
    sim_start_time: int = 4*60,
    sim_step_time: int = 60,
    sim_end_time: int = 24*60,
    num_simulations: int = 100,
    num_processes: int = 3
):
    """
        Parameters
        ----------
            in_dir_path: str
                (e.g. '../experiments/input_data/base_distributions')
                This folder should contain following subfolders
                and files in these subfolders:
                    - /demography/
                        - population_dist.json
                        - demography_dist.json
                    - /decision_tree/
                        - pub_trans_comfort_dist.json
                        - pub_trans_punctuality_dist.json
                        - bicycle_infrastr_comfort_dist.json
                        - pedestrian_inconvenience_dist.json
                        - household_persons_dist.json
                        - household_cars_dist.json
                        - household_bicycles_dist.json
                    - /travel_planning/
                        - any_travel_dist.json
                        - travel_chains_dist.json
                        - start_hour_dist.json
                        - other_travels_dist.json
                        - spend_time_dist_params.json
                        - trip_cancel_prob.json
                        - gravity_dist.json
                        - drivers_dist.json
            out_dir_path: str
                (e.g. '../experiments/results/base_distributions')
                If this folder does not exist, it will be created.
                In it, files will be created:
                    - 'agents_results_<sim_num>.csv'
                    - 'travels_results_<sim_num>.csv'
                Where <sim_num> means the simulation number and the number
                of such files depends on the parameter: num_simulations.
    """

    def flatten_dist(data,prev_keys=list(), sep="", lvl=0, max_level=1, normal=False):
        if lvl == max_level:
            return [
                (sep.join(prev_keys),
                ((np.array(list(data.keys())), np.array(list(data.values())))) if not normal else data)
            ]
        return [
            (flat_key, flay_val) 
            for key, value in data.items() 
            for flat_key, flay_val in flatten_dist(value, [*prev_keys, key], sep, lvl + 1, max_level, normal)
        ]

    def load_dist(name, in_dir='out'):
        file_name = name if name.endswith('.json') else (name + '.json')
        file_path = os.path.join(in_dir, file_name)

        with open(file_path, 'r') as f:
            return json.load(f)
            

    # Demography distributions
    data_dir = in_dir_path + '/demography/'

    data_file = 'population_dist.json'
    population_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir), max_level=0)[0][1]

    data_file = 'demography_dist.json'
    demography_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir), max_level=0)[0][1]

    # Decision tree distributions
    data_dir = in_dir_path + '/decision_tree/'

    data_file = 'pub_trans_comfort_dist.json'
    pub_trans_comfort_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'pub_trans_punctuality_dist.json'
    pub_trans_punctuality_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'bicycle_infrastr_comfort_dist.json'
    bicycle_infrastr_comfort_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'pedestrian_inconvenience_dist.json'
    pedestrian_inconvenience_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'household_persons_dist.json'
    household_persons_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'household_cars_dist.json'
    household_cars_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'household_bicycles_dist.json'
    household_bicycles_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    # Travel planning distributions
    data_dir = in_dir_path + '/travel_planning/'

    data_file = 'any_travel_dist.json'
    any_travel_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'travel_chains_dist.json'
    travel_chains_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'start_hour_dist.json'
    start_hour_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'other_travels_dist.json'
    other_travels_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    data_file = 'spend_time_dist_params.json'
    spend_time_dist_params = flatten_dist(load_dist(name=data_file, in_dir=data_dir), max_level=2, normal=True)

    data_file = 'trip_cancel_prob.json'
    trip_cancel_prob = load_dist(name=data_file, in_dir=data_dir)

    data_file = 'gravity_dist.json'
    gravity_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir), max_level=2)

    data_file = 'drivers_dist.json'
    drivers_dist = flatten_dist(load_dist(name=data_file, in_dir=data_dir))

    # Interregional distances and decision tree classifier
    data_dir = in_dir_path.replace(in_dir_path.split('/')[-1], '')

    # interregional distances
    data_file = 'interregional_distances.json'
    interregional_distances = load_dist(name=data_file, in_dir=data_dir)

    # decision tree
    data_file = 'decision_tree.pickle'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'rb') as f:
        decision_tree = pickle.load(f)

    params = {
        'N': num_agents,
        'population_dist': population_dist,
        'demography_dist': demography_dist,
        'pub_trans_comfort_dist': pub_trans_comfort_dist,
        'pub_trans_punctuality_dist': pub_trans_punctuality_dist,
        'bicycle_infrastr_comfort_dist': bicycle_infrastr_comfort_dist,
        'pedestrian_inconvenience_dist': pedestrian_inconvenience_dist,
        'household_persons_dist': household_persons_dist,
        'household_cars_dist': household_cars_dist,
        'household_bicycles_dist': household_bicycles_dist,
        'any_travel_dist': any_travel_dist,
        'travel_chains_dist': travel_chains_dist,
        'start_hour_dist': start_hour_dist,
        'other_travels_dist': other_travels_dist,
        'spend_time_dist_params': spend_time_dist_params,
        'trip_cancel_prob': trip_cancel_prob,
        'decision_tree': decision_tree,
        'gravity_dist': gravity_dist,
        'drivers_dist': drivers_dist,
        'interregional_distances': interregional_distances,
        'start_time': sim_start_time,
        'step_time': sim_step_time,
        'end_time': sim_end_time,
    }

    with Pool(num_processes) as p:
        p.map(
            run_single,
            [[params, i+1, out_dir_path] for i in range(num_simulations)]
        )


def run_single(params):
    model_params, run_num, out_dir_path = params
    model = TrafficModel(**model_params)

    for _ in range(
        model_params['start_time'],
        model_params['end_time']+1,
        model_params['step_time']
    ):
        model.step()

    agents_results = model.agent_data_collector.get_agent_vars_dataframe()
    travels_results = model.travels_data_collector.get_agent_vars_dataframe()
    travels_results = explode(
        travels_results[travels_results['start_region'].astype(str) != '[]'],
        [
            'start_region', 'start_place_type', 'dest_region',
            'dest_place_type', 'travel_start_time', 'dest_activity_dur_time',
            'transport_mode', 'is_driver'
        ],
        fill_value=''
    )

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    file_name = 'agents_results_' + str(run_num) + '.pkl'
    out_file = os.path.join(out_dir_path, file_name)
    agents_results.to_pickle(out_file)

    file_name = 'travels_results_' + str(run_num) + '.pkl'
    out_file = os.path.join(out_dir_path, file_name)
    travels_results.to_pickle(out_file)
