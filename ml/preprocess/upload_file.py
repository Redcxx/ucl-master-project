from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.settings import LoadSettingsFile

print(f'Connecting to Google Drive for Saving and Backup')
g_auth = GoogleAuth(settings_file='ucl-master-project/misc/settings.yaml')
# pydrive2 swallow error, we load it again to ensure it really works
print('Loading settings file ... ', end='')
LoadSettingsFile('ucl-master-project/misc/settings.yaml')
print('done')
g_auth.CommandLineAuth()
drive = GoogleDrive(g_auth)
print(f'Authentication finished')

import os

os.chdir('ucl-master-project')
from ml.save_load import ensure_folder_on_drive

os.chdir('../../gitlab')

working_folder = ensure_folder_on_drive(drive, 'WORK')


def upload_file(file_path):
    # ensure local has it
    if not os.path.isfile(file_path):
        print(f'file: {file_path} does not exists, not uploading')
        return

    files_on_drive = drive.ListFile({
        # see https://developers.google.com/drive/api/guides/search-files
    }).GetList()

    def match_parent(parents):
        return any(parent['id'] == working_folder['id'] for parent in parents)

    file_name = os.path.basename(os.path.normpath(file_path))
    files_on_drive = [file for file in files_on_drive if file['title'] == file_name and match_parent(file['parents'])]

    if len(files_on_drive) != 0:
        print('File found on drive, not uploading:')
        print(files_on_drive)
        return

    file = drive.CreateFile({
        'title': file_name,
        'parents': [{
            'id': working_folder['id']
        }]
    })
    file.SetContentFile(file_path)
    # save to google drive
    print('Uploading')
    file.Upload()


upload_file('alacgan_colorization_data.zip')
