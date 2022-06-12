import os
import subprocess
import sys
import time
from pprint import pprint

import torch
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from torch import optim

from ml.session import SessionOptions

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import files, drive

    drive_dir = '/content/drive'
    drive.mount(drive_dir)

    working_dir = os.path.join(drive_dir, 'My Drive', sconfig.working_folder, sconfig.run_id)
    print(working_dir)
    Path(working_dir).mkdir(parents=True, exist_ok=True)  # create directory if not exists on google drive


    def save_file(file_name, local=True):
        # save locally
        if local:
            files.download(file_name)

            # save on google drive
        with open(file_name, 'rb') as src_file:
            with open(os.path.join(working_dir, file_name), 'wb') as dest_file:
                dest_file.write(src_file.read())


    def load_file(file_name):
        if os.path.isfile(file_name):
            print(f'"{file_name}" already exists, not downloading')
            return True
        exit_code = subprocess.call(f'cp "{os.path.join(working_dir, file_name)}" "{file_name}"')
        if exit_code is None:
            # did not terminate
            return False
        return exit_code >= 0


else:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive


    def ensure_folder_on_drive(drive, folder_name):
        folders = drive.ListFile({
            # see https://developers.google.com/drive/api/guides/search-files
            'q': "mimeType = 'application/vnd.google-apps.folder'"
        }).GetList()

        folders = list(filter(lambda folder: folder['title'] == folder_name, folders))

        if len(folders) == 1:
            return folders[0]

        if len(folders) > 1:
            pprint(folders)
            raise AssertionError('Multiple Folders of the same name detected')

        # folder not found, create a new one at root
        print(f'Folder: {folder_name} not found, creating at root')

        folder = drive.CreateFile({
            'title': folder_name,
            # "parents": [{
            #     "kind": "drive#fileLink",
            #     "id": parent_folder_id
            # }],
            "mimeType": "application/vnd.google-apps.folder"
        })
        folder.Upload()
        return folder


    g_auth = GoogleAuth(settings_file=sconfig.pydrive2_setting_file, http_timeout=None)
    g_auth.LocalWebserverAuth(host_name="localhost", port_numbers=None, launch_browser=True)
    drive = GoogleDrive(g_auth)

    folder = ensure_folder_on_drive(drive, sconfig.working_folder)


    def save_file(file_name, local=True):
        file = drive.CreateFile({
            'title': file_name,
            'parents': [{
                'id': folder['id']
            }]
        })
        file.SetContentFile(file_name)
        # save to google drive
        file.Upload()
        # save locally
        if local:
            file.GetContentFile(file_name)


    def load_file(file_name):
        if os.path.isfile(file_name):
            print(f'"{file_name}" already exists, not downloading')
            return True
        files = drive.ListFile({
            'q': f"'{folder['id']}' in parents"
        }).GetList()
        for file in files:
            if file['title'] == file_name:
                # download
                drive.CreateFile({'id': file['id']}).GetContentFile(file_name)
                return True
        return False  # no match file


def format_time(seconds):
    return time.strftime('%Hh:%Mm:%Ss', time.gmtime(seconds))
