import json
import os
import pickle
from multiprocessing import Pool

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
                        - travels_num_dist.json
                        - start_hour_dist.json
                        - dest_type_dist.json
                        - spend_time_dist_params.json
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

    # Demography distributions
    data_dir = in_dir_path + '/demography/'

    data_file = 'population_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        population_dist = json.load(f)

    data_file = 'demography_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        demography_dist = json.load(f)

    # Decision tree distributions
    data_dir = in_dir_path + '/decision_tree/'

    data_file = 'pub_trans_comfort_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        pub_trans_comfort_dist = json.load(f)

    data_file = 'pub_trans_punctuality_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        pub_trans_punctuality_dist = json.load(f)

    data_file = 'bicycle_infrastr_comfort_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        bicycle_infrastr_comfort_dist = json.load(f)

    data_file = 'pedestrian_inconvenience_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        pedestrian_inconvenience_dist = json.load(f)

    data_file = 'household_persons_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        household_persons_dist = json.load(f)

    data_file = 'household_cars_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        household_cars_dist = json.load(f)

    data_file = 'household_bicycles_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        household_bicycles_dist = json.load(f)

    # Travel planning distributions
    data_dir = in_dir_path + '/travel_planning/'

    data_file = 'travels_num_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        travels_num_dist = json.load(f)

    data_file = 'start_hour_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        start_hour_dist = json.load(f)

    data_file = 'dest_type_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        dest_type_dist = json.load(f)

    data_file = 'spend_time_dist_params.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        spend_time_dist_params = json.load(f)

    data_file = 'gravity_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        gravity_dist = json.load(f)

    data_file = 'drivers_dist.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        drivers_dist = json.load(f)

    # Interregional distances and decision tree classifier
    data_dir = '../experiments/input_data'

    # interregional distances
    data_file = 'interregional_distances.json'
    data_path = os.path.join(data_dir, data_file)
    with open(data_path, 'r') as f:
        interregional_distances = json.load(f)

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
        'travels_num_dist': travels_num_dist,
        'start_hour_dist': start_hour_dist,
        'dest_type_dist': dest_type_dist,
        'spend_time_dist_params': spend_time_dist_params,
        'decision_tree': decision_tree,
        'gravity_dist': gravity_dist,
        'drivers_dist': drivers_dist,
        'interregional_distances': interregional_distances,
        'start_time': sim_start_time,
        'step_time': sim_step_time,
        'end_time': sim_end_time,
        'out_dir_path': out_dir_path
    }

    with Pool(num_processes) as p:
        p.map(
            run_single,
            [[params, i+1] for i in range(num_simulations)]
        )


def run_single(params):

    run_num = params[1]
    params = params[0]

    model = TrafficModel(
        N=params['N'],
        population_dist=params['population_dist'],
        demography_dist=params['demography_dist'],
        pub_trans_comfort_dist=params['pub_trans_comfort_dist'],
        pub_trans_punctuality_dist=params['pub_trans_punctuality_dist'],
        bicycle_infrastr_comfort_dist=params['bicycle_infrastr_comfort_dist'],
        pedestrian_inconvenience_dist=params['pedestrian_inconvenience_dist'],
        household_persons_dist=params['household_persons_dist'],
        household_cars_dist=params['household_cars_dist'],
        household_bicycles_dist=params['household_bicycles_dist'],
        travels_num_dist=params['travels_num_dist'],
        start_hour_dist=params['start_hour_dist'],
        dest_type_dist=params['dest_type_dist'],
        spend_time_dist_params=params['spend_time_dist_params'],
        decision_tree=params['decision_tree'],
        gravity_dist=params['gravity_dist'],
        drivers_dist=params['drivers_dist'],
        interregional_distances=params['interregional_distances'],
        start_time=params['start_time'],
        step_time=params['step_time'],
        end_time=params['end_time']
    )

    for i in range(
        params['start_time'],
        params['end_time']+1,
        params['step_time']
    ):
        model.step()

    agents_results = model.agent_data_collector.get_agent_vars_dataframe()
    travels_results = model.travels_data_collector.get_agent_vars_dataframe()
    travels_results = explode(
        travels_results[travels_results['start_region'].astype(str) != '[]'],
        [
            'start_region', 'start_place_type', 'dest_region',
            'dest_place_type', 'travel_start_time', 'transport_mode',
            'is_driver'
        ],
        fill_value=''
    )

    out_dir_path = params['out_dir_path']
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    file_name = 'agents_results_' + str(run_num) + '.csv'
    out_file = os.path.join(out_dir_path, file_name)
    agents_results.to_csv(out_file)

    file_name = 'travels_results_' + str(run_num) + '.csv'
    out_file = os.path.join(out_dir_path, file_name)
    travels_results.to_csv(out_file)
