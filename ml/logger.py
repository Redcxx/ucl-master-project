import os

LOG_FILE = 'log.txt'


def log(text, save_local=True, end=os.linesep):
    text = str(text) + str(end)
    if save_local:
        with open(LOG_FILE, 'a') as file:
            file.write(text)
    print(text)
