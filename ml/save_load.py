import os
import subprocess
import time
from pathlib import Path

from google.colab import files, drive  # only works in google colab

drive_dir = '/content/drive'
drive.mount(drive_dir)

WORKING_DIR = ''


def init(opt):
    global WORKING_DIR
    if not WORKING_DIR:
        WORKING_DIR = os.path.join(drive_dir, 'My Drive', opt.working_folder, opt.run_id)
        print(WORKING_DIR)
        Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)  # create directory if not exists on google drive


def save_file(opt, file_name, local=True):
    init(opt)
    # save locally
    if local:
        files.download(file_name)

        # save on google drive
    with open(file_name, 'rb') as src_file:
        with open(os.path.join(WORKING_DIR, file_name), 'wb') as dest_file:
            dest_file.write(src_file.read())


def load_file(opt, file_name):
    init(opt)
    if os.path.isfile(file_name):
        print(f'"{file_name}" already exists, not downloading')
        return True
    exit_code = subprocess.call(f'cp "{os.path.join(WORKING_DIR, file_name)}" "{file_name}"')
    if exit_code is None:
        # did not terminate
        return False
    return exit_code >= 0


def format_time(seconds):
    return time.strftime('%Hh:%Mm:%Ss', time.gmtime(seconds))
