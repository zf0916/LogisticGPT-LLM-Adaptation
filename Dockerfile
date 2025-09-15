FROM python:3.11-slim

# Create user and set up environment
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    libpq-dev

USER appuser
WORKDIR /home/appuser

# Copy application files
COPY . app

# Set up virtual environment and install requirements
ENV VIRTUAL_ENV=/home/appuser/venv
RUN virtualenv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install .
EXPOSE 8501

WORKDIR /home/appuser/app

USER root

RUN chmod +x /home/appuser/app/run.sh
RUN chown -R appuser:appuser /home/appuser/app
RUN chown -R appuser:appuser /home/appuser
USER appuser

ENTRYPOINT ["./run.sh"]