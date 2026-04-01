FROM ubuntu

RUN apt-get update && apt-get install -y \
    nano \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pandas --break-system-packages

ENTRYPOINT ["tail", "-f", "/dev/null"]
