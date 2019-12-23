FROM anibali/pytorch:latest
SHELL ["/bin/bash", "-c"]

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG TWILIO_AUTH_TOKEN
ARG TWILIO_PHONE
ARG MYPHONE
ARG TWILIO_AUTH_TOKEN
ARG TWILIO_ACCOUNT_SID
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION="us-east-1"
ENV USE_GPU=1
ENV TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID
ENV TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN
ENV TWILIO_PHONE=$TWILIO_PHONE
ENV MYPHONE=$MYPHONE

USER root
RUN apt update
RUN apt-get install -y build-essential
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y vim

WORKDIR /app
RUN conda create -n venv python=3.6
RUN source activate venv
# we need this specific pytorch version for nonechucks
RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

COPY src /app/src
COPY config /app/config
COPY setup.py /app/setup.py
COPY requirements.txt /app/requirements.txt

RUN mkdir models
RUN mkdir submissions

RUN pip install -r /app/requirements.txt
RUN pip install .
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8