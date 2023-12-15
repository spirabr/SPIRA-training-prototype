# Use PyTorch as base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel as base

# Default work directory
WORKDIR /app

# Build arguments
ARG PIP_VERSION=23.3.1
ARG POETRY_VERSION=1.5.1
ARG WHEN_CHANGED_VERSION=0.3.0

# Run inside poetry environment by default
ENTRYPOINT ["poetry", "run", "--"]

# Install poetry version
RUN pip install pip==${PIP_VERSION} \
    poetry==${POETRY_VERSION} \
    when-changed==${WHEN_CHANGED_VERSION}

# Copy poetry-managed dependencies
COPY poetry.toml pyproject.toml poetry.lock ./

# Install dependencies
# (SPIRA code not available yet, so it won't be installed)
RUN poetry install

# Copy SPIRA code
COPY . .

# Install SPIRA code
# (SPIRA code is made available inside poetry's venv)
RUN poetry install

# Run SPIRA module (installed in poetry's venv)
CMD ["python", "-m", "spira.tasks.train"]
