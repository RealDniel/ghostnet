FROM python:3.10-slim
WORKDIR /app
COPY hpc_train.py .
RUN pip install --no-cache-dir \
    "tensorflow==2.17.0" \
    snowflake-connector-python \
    scikit-learn \
    scipy \
    numpy
CMD ["python3", "-u", "hpc_train.py"]
