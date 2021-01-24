SERVICES_NUMBER=$1
RESULTS_PATH=$2
POPULATION=$3
NUM_SIMULATIONS=$4
NUM_PROCESSES=$5

python3 ./experiments/prepare_scenarios.py [decision_tree/pub_trans_comfort_dist-up,decision_tree/pub_trans_punctuality_dist-up,decision_tree/household_cars_dist-down] [0.,0.05,0.1,0.15] $SERVICES_NUMBER ./experiments/scenarios

# docker build . -t mobility-simulator

# for i in $(seq 1 $SERVICES_NUMBER);
# do
# docker run --mount src=$RESULTS_PATH,target=/mobility-project/results,type=bind -d mobility-simulator python3 run_simulations.py /mobility-project/scenarios scenarios_$i /mobility-project/input_data /mobility-project/results $POPULATION $NUM_SIMULATIONS $NUM_PROCESSES;
# done
