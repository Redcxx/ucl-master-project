import logging
import os
from pathlib import Path

LOG_FILE = 'log.txt'
FIRST_TIME = True


def log(text, save_local=True, end=os.linesep):
    text = str(text) + str(end)

    print(text, end='')

    if save_local:
        global FIRST_TIME
        if FIRST_TIME:
            open(LOG_FILE, 'w').close()  # create file and erase content
            FIRST_TIME = False
        with open(LOG_FILE, 'a') as file:
            file.write(text)
