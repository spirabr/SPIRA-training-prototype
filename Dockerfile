# --- base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel as base

WORKDIR /app

ENV POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false

EXPOSE 8000

ENTRYPOINT ["python"]

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

# --no-root : https://stackoverflow.com/questions/75397736/poetry-install-on-an-existing-project-error-does-not-contain-any-element
RUN poetry install --no-root

COPY . .

CMD ["training-pipeline/model_pipeline.py"]
