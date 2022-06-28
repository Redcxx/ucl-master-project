import os

LOG_FILE = 'log.txt'


def log(text, save_local=True, end=os.pathsep):
    text += end
    print(text)
    if save_local:
        with open(LOG_FILE, 'a') as file:
            file.write(text)
