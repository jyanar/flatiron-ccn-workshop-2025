FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      'python3.11' \
      'python3.11-venv' \
      git \
      curl \
      pandoc \
      jq \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN python3.11 -m venv --system-site-packages /venv
COPY . /ccn-software-jan-2025/
# necessary for the pip install command to succeed
RUN mkdir /ccn-software-jan-2025/src
RUN . /venv/bin/activate && pip install /ccn-software-jan-2025/
