FROM python:3.8

WORKDIR /mobility-project

COPY requirements.txt .
COPY src /mobility-project/src
COPY setup.py /mobility-project/
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY experiments/input_data /mobility-project/input_data
COPY experiments/scenarios /mobility-project/scenarios
COPY experiments/run_simulations.py /mobility-project/

RUN mkdir /mobility-project/results