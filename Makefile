DOCKER_COMPOSE_DEV := docker compose --profile dev
DOCKER_COMPOSE_DEV_BUILD := ${DOCKER_COMPOSE_DEV} build
DOCKER_COMPOSE_DEV_START := ${DOCKER_COMPOSE_DEV} run --rm trainer
DOCKER_COMPOSE_DEV_RUN := ${DOCKER_COMPOSE_DEV} run --rm --entrypoint '/bin/bash -c' trainer
DOCKER_COMPOSE_DEV_SHELL := ${DOCKER_COMPOSE_DEV} run --rm --entrypoint '/bin/bash' trainer

all: build start

build:
	${DOCKER_COMPOSE_DEV_BUILD}

start:
	${DOCKER_COMPOSE_DEV_START}

black:
	${DOCKER_COMPOSE_DEV_RUN} "black ."

add-dependency:
	@echo "Installing dependency ${DEPENDENCY}"
	@${DOCKER_COMPOSE_DEV_RUN} "poetry add ${DEPENDENCY}"

terminal:
	${DOCKER_COMPOSE_DEV_SHELL}