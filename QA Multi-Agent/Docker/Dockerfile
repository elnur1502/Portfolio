# commented sections are necessary for my current workload. You can easily ignore this if you are not using Impala
# do not forget to install required libraries for your setup

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    python3-pip \
    curl \
    sudo \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8
   
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
RUN sudo apt install nodejs
RUN npm install -g @playwright/test
RUN npx playwright install
RUN npx playwright install-deps chrome
RUN npx playwright install-deps chromium
RUN npm install @playwright/mcp

COPY ./Docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
