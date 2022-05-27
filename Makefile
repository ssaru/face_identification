base_image=nvidia/cuda:11.3.0-runtime-ubuntu20.04
python_ver=3.8.5
image_name=pytorch-face-identification:v0
container_name=pytorch-face-identification

dev-env: set-git set-dep
test: pytest
build: bulid-docker
clean: clean-pyc clean-test

#### dev-env ####
set-git:
	git config --local commit.template .gitmessage.txt

set-dep:
	poetry install

set-precommit:
	pre-commit install

#### test #####
pytest: # poetry run pytest -n 1 -o log_cli=true --disable-pytest-warnings --cov-report term-missing tests/ 
	pytest -o log_cli=true -W ignore::DeprecationWarning --cov-report term-missing tests/*

check:
	pre-commit run -a

#### docker #####
build-docker:
	docker build -f docker/Dockerfile -t $(image_name) . --build-arg BASE_IMAGE=$(base_image) --build-arg PYTHON_VER=$(python_ver) --no-cache

run-docker:
	docker run -i -t -d --shm-size=8G --init --name $(container_name) $(image_name)

exec-docker:
	docker exec -it $(container_name) /bin/bash

rm-docker:
	docker stop $(container_name) && docker rm $(container_name)

####  clean  ####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf tests/output
