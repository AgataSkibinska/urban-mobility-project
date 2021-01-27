FROM python:3.7

WORKDIR /usr/src/app

COPY ./ ./
RUN pip install -r requirements.txt
EXPOSE 8501

#CMD [ "streamlit", "run", "./streamlit_dash.py" ]