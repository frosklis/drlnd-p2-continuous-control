FROM nvidia/cuda:10.2-base

# set a directory for the app
WORKDIR /usr/src/drlnd-p2-continuous

# copy all the files to the container
COPY . .

# install dependencies
RUN apt update && apt install python3-pip -y
RUN apt install wget -y
RUN apt install unzip -y
RUN apt install git -y
RUN pip3 install --no-cache-dir jupyter
RUN pip3 install --no-cache-dir -r requirements.txt
RUN cd extras && make Reacher_Linux_NoVis
# RUN cd /usr/src/drlnd-p2-continuous

# define the port number the container should expose
EXPOSE 8888

# run the command
# jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
# CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888"]