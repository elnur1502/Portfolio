# commented sections are necessary for my current workload. You can easily ignore this if you are not using Impala
# do not forget to install required libraries for your setup

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    # libsasl2-dev \
    # libsasl2-modules \
    python3-pip \
    # default-jre \
    && rm -rf /var/lib/apt/lists/* \
    # && sed -i 's/jdk.tls.disabledAlgorithms=.*/jdk.tls.disabledAlgorithms=SSLv3, RC4, DES, MD5withRSA/g' $(find /usr/lib/jvm -name java.security) \
    # && sed -i 's/jdk.certpath.disabledAlgorithms=.*/jdk.certpath.disabledAlgorithms=MD5, RSA keySize < 1024/g' $(find /usr/lib/jvm -name java.security)

RUN pip install --no-cache-dir numpy pandas matplotlib seaborn fastapi uvicorn jaydebeapi openpyxl

WORKDIR /app

COPY executor.py . 
#COPY ImpalaJDBC41.jar . # only if you work with Impala
#COPY truststore.jks .  # only if you work with Impala

CMD ["uvicorn", "executor:app", "--host", "0.0.0.0", "--port", "8000"]