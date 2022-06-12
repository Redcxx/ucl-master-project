import os
import time
from pprint import pprint

import torch
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from torch import optim

from ml.session import SessionOptions

_DRIVE_AND_FOLDER = None


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


def get_drive_and_folder(opt):
    global _DRIVE_AND_FOLDER
    pydrive2_setting_file = opt.pydrive2_setting_file
    working_folder = opt.working_folder

    if _DRIVE_AND_FOLDER is None:
        g_auth = GoogleAuth(settings_file=pydrive2_setting_file, http_timeout=None)
        g_auth.LocalWebserverAuth(host_name="localhost", port_numbers=None, launch_browser=True)
        drive = GoogleDrive(g_auth)

        folder = ensure_folder_on_drive(drive, working_folder)

        _DRIVE_AND_FOLDER = drive, folder

    return _DRIVE_AND_FOLDER


def save_file(opt: SessionOptions, file_name, local=True):
    drive, folder = get_drive_and_folder(opt)

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


def load_file(opt, file_name):
    drive, folder = get_drive_and_folder(opt)

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
