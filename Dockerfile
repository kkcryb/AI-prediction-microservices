FROM ubuntu:latest
LABEL authors="rain"

ENTRYPOINT ["top", "-b"]