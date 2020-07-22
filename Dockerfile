FROM ubuntu:18.04

MAINTAINER aswin1906@gmail.com

RUN apt-get update && apt-get install -y nodejs npm git

RUN apt-get update && apt-get install -y tmux &&\
    apt-get update &&  apt-get install -y curl wget nano vim

RUN apt-get update && apt-get install -y build-essential libssl-dev && apt install -y graphviz

# install miniconda
ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

RUN conda create --name py35 python=3.6

WORKDIR /app/ai-cnn

COPY . /app/ai-cnn

VOLUME . /app/ai-cnn

RUN conda init bash

RUN pip install -r requirements.txt

EXPOSE 8087

# Run the specified command within the container.
CMD [ "python manage.py runserver 0.0.0.0:8087" ]

# Copy the rest of your app's source code from your host to your image filesystem.
COPY . .
