FROM python:3.7

WORKDIR /usr/src/app

RUN pip install -r requirements.txt
COPY ./dashboard ./
EXPOSE 8501

#CMD [ "streamlit", "run", "./streamlit_dash.py" ]