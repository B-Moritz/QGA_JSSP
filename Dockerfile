# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.9.13
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /QGA_JSSP

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    qmea

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
# Note that this install ignores packages that failed to be installed
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    while read p; do pip install "$p"; done <requirements.txt

# Switch to the non-privileged user to run the application.
USER qmea

# Copy the source code into the container.
COPY . .

#RUN sudo chmod 700 run_current_experiment.bash
# Expose the port that the application listens on.
#EXPOSE 8000
USER root
RUN python install.py
RUN chown -R qmea /QGA_JSSP
RUN chmod 755 run_*
#RUN echo "/app/qga_lib" > /usr/local/lib/python3.9/site-packages/qga_lib.pth
USER qmea

# Run the application.
CMD ./run_current_experiment.bash
