import logging
import os
from pathlib import Path

LOG_FILE = 'log.txt'
FIRST_TIME = True


def log(text, save_local=True, end=os.linesep):
    text = str(text) + str(end)

    logging.info(text)

    if save_local:
        global FIRST_TIME
        if FIRST_TIME:
            path = Path(LOG_FILE)
            path.unlink(missing_ok=True)
            path.touch()
            FIRST_TIME = False
        with open(LOG_FILE, 'a') as file:
            file.write(text)
