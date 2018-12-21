FROM vanvalenlab/deepcell-tf:0.1

WORKDIR /kiosk/training

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["/bin/sh", "-c", "python train.py"]
