FROM python:3.12-slim

WORKDIR /work
COPY frontierscience_eval /work/frontierscience_eval
COPY configs /work/configs
COPY tests /work/tests

USER 65534:65534
CMD ["python", "-m", "frontierscience_eval", "--help"]
