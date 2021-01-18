import json
import os
import pickle

# ------------------------------------------------------------
## Load distributions

# ------------------------------------------------------------
# demography distributions
data_dir = 'input_data/base_distributions/demography/'

data_file = 'population_dist.json'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    population_dist = json.load(f)

data_file = 'demography_dist.json'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    demography_dist = json.load(f)


# decision tree distributions
data_dir = 'input_data/base_distributions/decision_tree/'

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


# travel planning distributions
data_dir = 'input_data/base_distributions/travel_planning/'

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

data_file = 'gravity_dist.json'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    gravity_dist = json.load(f)

data_file = 'drivers_dist.json'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    drivers_dist = json.load(f)

# ------------------------------------------------------------
## Load interregional distances and decision tree classifier

# ------------------------------------------------------------
data_dir = 'input_data/'

# interregional distances
data_file = 'interregional_distances.json'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    interregional_distances = json.load(f)


# decision tree
data_file = 'decision_tree.pickle'
data_path = os.path.join(data_dir, data_file)
with open(data_path, 'r') as f:
    decision_tree = pickle.load(f)

# ------------------------------------------------------------
## TrafficModel test

# ------------------------------------------------------------
start_time = 4 * 60
step_time = 60
end_time = 23 * 60


model = TrafficModel(
    N=10000,  # 635701
    population_dist=population_dist,
    demography_dist=demography_dist,
    pub_trans_comfort_dist=pub_trans_comfort_dist,
    pub_trans_punctuality_dist=pub_trans_punctuality_dist,
    bicycle_infrastr_comfort_dist=bicycle_infrastr_comfort_dist,
    pedestrian_inconvenience_dist=pedestrian_inconvenience_dist,
    household_persons_dist=household_persons_dist,
    household_cars_dist=household_cars_dist,
    household_bicycles_dist=household_bicycles_dist,
    travels_num_dist=travels_num_dist,
    start_hour_dist=start_hour_dist,
    dest_type_dist=dest_type_dist,
    spend_time_dist_params=spend_time_dist_params,
    decision_tree=decision_tree,
    gravity_dist=gravity_dist,
    drivers_dist=drivers_dist,
    interregional_distances=interregional_distances,
    start_time=start_time,
    step_time=step_time
)

for i in tqdm(range(start_time, end_time+1, step_time)):
    model.step()