FROM python:3.8-slim

COPY . /project

WORKDIR /project

RUN apt-get update && apt-get install -y python3-pip


RUN pip3 install -r requirements.txt 

EXPOSE 8000

CMD [ "python3", "src/app/main.py" ]