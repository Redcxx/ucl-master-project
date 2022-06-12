import os
import time
from pprint import pprint

import torch
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from torch import optim

from ml.models import Generator, Discriminator


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


def init_google_drive(pydrive2_setting_file, working_folder):
    g_auth = GoogleAuth(settings_file=pydrive2_setting_file, http_timeout=None)
    g_auth.LocalWebserverAuth(host_name="localhost", port_numbers=None, launch_browser=True)
    drive = GoogleDrive(g_auth)

    folder = ensure_folder_on_drive(drive, working_folder)

    return drive, folder


def save_file(drive, folder, file_name, local=True):
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


def load_file(drive, folder, file_name):
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


def save_checkpoint(sconfig, net_G, net_D, optimizer_G, optimizer_D, tag=''):
    file_name = f'{sconfig.run_id}{tag}.ckpt'
    torch.save({
        'net_G_state_dict': net_G.state_dict(),
        'net_D_state_dict': net_D.state_dict(),
        'net_G_optimizer_state_dict': optimizer_G.state_dict(),
        'net_D_optimizer_state_dict': optimizer_D.state_dict(),
        'session_config': sconfig
    }, file_name)
    save_file(file_name, local=False)
    return file_name


def load_checkpoint(run_id, device, tag=''):
    file_name = f'{run_id}{tag}.ckpt'
    load_file(file_name)  # ensure exists locally, will raise error if not exists
    checkpoint = torch.load(file_name)

    loaded_config = checkpoint['session_config']

    net_G = Generator(loaded_config.generator_config)
    net_D = Discriminator(loaded_config.discriminator_config)
    net_G.load_state_dict(checkpoint['net_G_state_dict'])
    net_D.load_state_dict(checkpoint['net_D_state_dict'])
    net_G.to(device)
    net_D.to(device)

    optimizer_G = optim.Adam(net_G.parameters(), lr=loaded_config.lr,
                             betas=(loaded_config.optimizer_beta1, loaded_config.optimizer_beta2))
    optimizer_D = optim.Adam(net_D.parameters(), lr=loaded_config.lr,
                             betas=(loaded_config.optimizer_beta1, loaded_config.optimizer_beta2))
    optimizer_G.load_state_dict(checkpoint['net_G_optimizer_state_dict'])
    optimizer_D.load_state_dict(checkpoint['net_D_optimizer_state_dict'])

    return loaded_config, net_G, net_D, optimizer_G, optimizer_D



