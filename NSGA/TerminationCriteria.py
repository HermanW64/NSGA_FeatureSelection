from pymoo.termination import get_termination
import logging
logging.basicConfig(level="INFO")
"""
set termination criteria for the NSGA
value of "n_gen": max number of generation
"""


def termination_criteria(max_gen=100):
    termination = get_termination("n_gen", max_gen)
    logging.info("termination criteria set!")

    return termination
