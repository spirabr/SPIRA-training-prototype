DOCKER_COMPOSE_DEV := docker compose --profile dev
DOCKER_COMPOSE_DEV_BUILD := ${DOCKER_COMPOSE_DEV} build
DOCKER_COMPOSE_DEV_UP := ${DOCKER_COMPOSE_DEV} up
DOCKER_COMPOSE_DEV_RUN := ${DOCKER_COMPOSE_DEV} run --entrypoint '/bin/bash -c' --rm trainer

all: build start

build:
	${DOCKER_COMPOSE_DEV_BUILD}

start:
	${DOCKER_COMPOSE_DEV_UP}

add-dependency:
	@echo "Installing dependency ${DEPENDENCY}"
	@${DOCKER_COMPOSE_DEV_RUN} "poetry add ${DEPENDENCY}"