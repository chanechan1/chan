FROM python:3.9
#ENV PYTHONBUFFERED=1

ARG GIT_ID
ARG GIT_PASSWORD
ARG GIT_USERNAME
ARG GIT_EMAIL

# install ssh
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y openssh-server
RUN apt-get install -y libeccodes0

# install vim
RUN apt-get install -y vim wget xvfb unzip  gnupg gnupg2 gnupg1 python3-pip


# install src
WORKDIR /home
RUN git clone https://$GIT_ID:$GIT_PASSWORD@github.com/hellokeanu/advanced.git
RUN git config --global user.name "$GIT_USERNAME"
RUN git config --global user.email "$GIT_EMAIL"


# Put the github depositary name
WORKDIR /home/advanced
RUN pip install --upgrade pip --user
RUN pip install  --user numpy pandas APScheduler requests psycopg2 sqlalchemy openpyxl lxml selenium  netCDF4
RUN pip install  --user zarr cfgrib scipy eccodes
RUN pip install  --user xarray
RUN pip install  --user matplotlib scikit-learn joblib paramiko

ENTRYPOINT ["sh", "/home/advanced/entrypoint.sh" ]


