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


def download_file(file_name):
    if os.path.isfile(file_name):
        print(f'file: {file_name} already exists, not downloading')
    else:
        print(f'file: {file_name} does not exists, searching for it on drive')
        files = drive.ListFile({
            'q': f"'{working_folder['id']}' in parents"
        }).GetList()

        for file in files:
            if file['title'] == file_name:
                # download
                print('downloading file requested')
                drive.CreateFile({'id': file['id']}).GetContentFile(file_name)
                print('Finished')
                print()
                break


download_file('alacgan_colorization_data.zip')
download_file('vgg16-397923af.pth')
download_file('i2v.pth')
