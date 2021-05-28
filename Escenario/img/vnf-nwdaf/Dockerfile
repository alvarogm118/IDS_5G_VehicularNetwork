FROM ubuntu:bionic
# install required packages
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get clean
RUN apt-get update \
    && apt-get install -y  git \
	wget \
	unzip \
    net-tools \
    aptitude \
    build-essential \
	python3-pip \
    python3-setuptools \
    python3-dev \
    software-properties-common \
    ansible \
    curl \
    iptables \
    iputils-ping \
	iproute2 \
    sudo \
    nano \
    bridge-utils \
    isc-dhcp-server \
    tcpdump \
    openvswitch-switch \
    iperf3 \
    vim

#Machine Learning tools
RUN pip3 install --upgrade pip
RUN pip3 install wrapt --upgrade --ignore-installed
RUN pip3 install --upgrade tensorflow
RUN pip3 install keras 
RUN pip3 install matplotlib
RUN	pip3 install numpy==1.16.4
RUN	pip3 install pandas 
RUN	pip3 install sklearn 
RUN	pip3 install seaborn 
RUN	pip3 install pyyaml 
RUN	pip3 install h5py 
RUN pip3 install silence_tensorflow


COPY dhcpd.conf /etc/dhcp/dhcpd.conf
COPY isc-dhcp-server /etc/default/isc-dhcp-server

COPY IDS_REAL.py IDS_REAL.py
COPY model.h5 model.h5
COPY OBU_data1.csv OBU_data1.csv
COPY OBU_data2.csv OBU_data2.csv
COPY OBU_data3.csv OBU_data3.csv
COPY OBU_data4.csv OBU_data4.csv

#Install isc-dhcp-server
#RUN apt-get update
#RUN apt-get install isc-dhcp-server 
#RUN apt-get update
#RUN apt-get install isc-dhcp-relay

#ENTRYPOINT python3 IDS_REAL.py

# open ports
# IPERF3
EXPOSE 5201
# IPERF3
EXPOSE 3000
# IPERF3
EXPOSE 7000

