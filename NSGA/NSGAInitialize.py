from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
import logging
logging.basicConfig(level="INFO")
"""
set hyperparameters for NSGA algorithm:
pop_size: # of individuals in the population
n_offsprings: # of offsprings

crossover: 
mutation:
"""


def set_NSGA():
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.01, eta=20),
        eliminate_duplicates=True
    )

    logging.info("NSGA parameters set!")

    return algorithm
