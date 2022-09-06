FROM python:3.8

ENV INSTALL_PATH /app

WORKDIR $INSTALL_PATH

# Install libs for opencv
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]

CMD ["app.py"]
