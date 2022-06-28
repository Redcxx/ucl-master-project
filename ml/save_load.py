import os
import pprint

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.settings import LoadSettingsFile

from ml.logger import log
from ml.options.base import BaseOptions

_DRIVE_AND_FOLDER = None


def ensure_folder_on_drive(drive, folder_name, parent_folder=None):
    folders = drive.ListFile({
        # see https://developers.google.com/drive/api/guides/search-files
        'q': "mimeType = 'application/vnd.google-apps.folder'"
    }).GetList()

    folders = list(filter(lambda f: f['title'] == folder_name, folders))

    if len(folders) == 1:
        return folders[0]

    if len(folders) > 1:
        log(pprint.pformat(folders))
        raise AssertionError('Multiple Folders of the same name detected')

    # folder not found, create a new one at root
    # pprint(parent_folder)
    log(f'Folder {folder_name} with parent {parent_folder["title"]} not found, creating... ', end='')

    folder = drive.CreateFile({
        'title': folder_name,
        "parents": [{
            "kind": "drive#fileLink",
            "id": parent_folder['id']
        }] if parent_folder else [],
        "mimeType": "application/vnd.google-apps.folder"
    })
    folder.Upload()

    log('done')
    return folder


def init_drive_and_folder(opt):
    global _DRIVE_AND_FOLDER

    if _DRIVE_AND_FOLDER is None:
        log(f'Connecting to Google Drive for Saving and Backup')
        g_auth = GoogleAuth(settings_file=opt.pydrive2_settings_file)
        # pydrive2 swallow error, we load it again to ensure it really works
        log('Loading settings file ... ', end='')
        LoadSettingsFile(opt.pydrive2_settings_file)
        log('done')
        # print('Save & Load Settings:')
        # print(g_auth.settings)
        # g_auth.LocalWebserverAuth()
        g_auth.CommandLineAuth()
        drive = GoogleDrive(g_auth)
        working_folder = ensure_folder_on_drive(drive, opt.working_folder)
        session_folder = ensure_folder_on_drive(drive, opt.run_id, parent_folder=working_folder)
        log(f'Authentication Finished')

        _DRIVE_AND_FOLDER = drive, session_folder

    return _DRIVE_AND_FOLDER


def save_file(opt: BaseOptions, file_name, local=True):
    drive, folder = init_drive_and_folder(opt)

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
    drive, folder = init_drive_and_folder(opt)

    if os.path.isfile(file_name):
        log(f'"{file_name}" already exists, not downloading')
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
