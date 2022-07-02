[![Python testing](https://github.com/alexarntzen/fordypningsprosjekt/workflows/Python%20testing/badge.svg)](https://github.com/alexarntzen/fordypningsprosjekt/actions/workflows/python_test.yml)
[![Python linting](https://github.com/alexarntzen/fordypningsprosjekt/workflows/Python%20linting/badge.svg)](https://github.com/alexarntzen/fordypningsprosjekt/actions/workflows/python_lint.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Nural reparametrization
This repo contains the code for my specialization project at NTNU and part of the code for my master thesis. This is because they are two ways of solving the same problem.


The experiments can be found in the `/experiments` folder. The `experiments/approx_phi` directory contains the experiments for the specialization project. In these examples a reparametrization is approximated directly. The `experiments/dqn` contains the experiments for the master thesis. In these examples the value function is approximated using deep Q-networks.

Also note that the code specifically for the specialization project is located in `neural_reparm/reparametrization.py`.

The code written specifically for the master thesis is located in two files. The code that implements the altered [DQN](https://doi.org/10.1038/nature14236) optimization algorithm is located in `neural_reparm/reinforcement_learning`. The code that implements the reparametrization environment is located in `neural_reparm/reparam_env`.

To run the first experiments, install the packages in `requirements.txt`. For instance with 

    pip3 install -r requirements.txt 

To run experiments using motion capture data. Set up the motion capture database as explained in [alexarntzen/signatureshape](https://github.com/alexarntzen/signatureshape). This is a copy of [paalel/Signatures-in-Shape-Analysis](https://github.com/paalel/Signatures-in-Shape-Analysis) that works with `python3`.