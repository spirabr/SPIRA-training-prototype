# SPIRA Training Pipeline

A training pipeline to [SPIRA](https://spira.ime.usp.br/) project. 

## Getting Started
### Prerequisites

What things you need to install

* [Docker](https://docs.docker.com/engine/install/)
* [Docker Compose](https://docs.docker.com/compose/install/linux/)

## Running the application

### Build Image

```bash
make build
```

### Running the container

```bash
make start
```

### Add a dependency for your code

Example
```bash
make add-dependency DEPENDENCY="pytorch==1.0.2"
```

## Built With

* [Python 3.10](https://docs.python.org/3.10/) - The language used
* [Poetry](https://python-poetry.org/) - Python Packaging and Dependency Management
* [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/linux/) - Containerization
* [GitHub](https://github.com/) - Version Control

## Versioning

[GitHub](https://github.com/) used for versioning. Version are available at [SPIRA-Training repository](https://github.com/spirabr/SPIRA-training). 

## Authors

* [**Daniel Angelo Esteves Lawand**](https://github.com/danlawand)

## Acknowledgments

* [Billie Thompson](https://purplebooth.co.uk/about/me) - To the [README template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [Renato Cordeiro Ferreira](https://linktr.ee/renatocf)
* [Vitor Daisuke Tamae](https://www.linkedin.com/in/vitor-tamae/) - To the [Spira Inference Service](https://github.com/spirabr/SPIRA-Inference-Service)


<!--
Acess container via command line:
docker compose --profile <profile_name> run --rm --entrypoint /bin/bash <service_name>
Write 'exit' to exit the container terminal.

Run a command inside the container:
docker compose --profile <profile_name> run --rm --entrypoint '/bin/bash -c' <service_name> '<command>'

Stop and remove containers 
docker compose down 


-->