from execution.RunOnSonar import run_on_sonar
from execution.RunOnArrhythmia import run_on_arrhythmia
import multiprocessing
import time
import logging
logging.basicConfig(level="INFO")


def main():
    process_1 = multiprocessing.Process(target=run_on_sonar)
    process_2 = multiprocessing.Process(target=run_on_arrhythmia)

    process_1.start()
    process_2.start()

    process_1.join()
    process_2.join()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    start_t = time.time()
    multiprocessing.freeze_support()

    main()

    end_t = time.time()
    total_s = end_t - start_t
    total_min = round(total_s / 60, 0)
    logging.info("=====In the End=====")
    logging.info("total time for execution: " + str(total_min) + " minutes.")

