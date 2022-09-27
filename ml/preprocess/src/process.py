import os
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import imagehash
from PIL import Image, ImageOps
from tqdm import tqdm

CROP_EMPTY = True
RESIZE_SIZE = 512

CROP_INSTEAD_OF_PAD = True
RETURN_HASH = False


# worker method
def process(paths):
    line_path, fill_path, save_path, i = paths

    # read image
    im_A = Image.open(line_path)
    im_B = Image.open(fill_path)

    if im_A.size != im_B.size:
        im_B = im_B.resize(im_A.size, resample=Image.BICUBIC)

    if CROP_EMPTY:
        # crop to non empty region
        non_empty_coord = im_A.getbbox()

        if non_empty_coord is None:
            # empty image
            return None, i

        im_A = im_A.crop(non_empty_coord)
        im_B = im_B.crop(non_empty_coord)

    if CROP_INSTEAD_OF_PAD:
        crop_size = min(im_A.size[0], im_A.size[1])
        crop_coord = center_crop_coord(im_A, (crop_size, crop_size))

        im_A = im_A.crop(crop_coord)
        im_B = im_B.crop(crop_coord)

        im_A = im_A.resize((RESIZE_SIZE, RESIZE_SIZE), resample=Image.BICUBIC)
        im_B = im_B.resize((RESIZE_SIZE, RESIZE_SIZE), resample=Image.BICUBIC)
    else:
        # pad and resize to square
        im_A = ImageOps.pad(im_A, (RESIZE_SIZE, RESIZE_SIZE), method=Image.BICUBIC, color='rgba(0,0,0,0)')
        im_B = ImageOps.pad(im_B, (RESIZE_SIZE, RESIZE_SIZE), method=Image.BICUBIC, color='rgba(0,0,0,0)')

    # combine
    im = Image.new('RGBA', (RESIZE_SIZE * 2, RESIZE_SIZE))
    im.paste(im_A, (0, 0))
    im.paste(im_B, (RESIZE_SIZE, 0))

    # convert to RGB, do not use .convert('RGB') as it gives black background
    rgb = Image.new("RGB", im.size, (255, 255, 255))
    rgb.paste(im, mask=im.split()[3])

    rgb.save(save_path)

    if RETURN_HASH:
        return imagehash.phash(rgb), i
    else:
        return i


def center_crop_coord(im, new_size):
    width, height = im.size
    new_width, new_height = new_size

    assert width >= new_width
    assert height >= new_height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return left, top, right, bottom


def _multi(Pool, amount, worker_inputs):
    seen = set()
    valids = 0
    dups = []
    empties = 0

    with Pool(amount) as pool:
        for im_hash, i in tqdm(pool.imap_unordered(process, worker_inputs), total=len(worker_inputs)):
            if im_hash is None:
                empties += 1
                continue
            else:
                valids += 1

                # check duplicates
                if im_hash in seen:
                    dups.append(worker_inputs[i][2])  # for later delete file
                else:
                    seen.add(im_hash)

    # remove duplicates
    for p in dups:
        Path(p).unlink(missing_ok=False)

    print(fr'Amount     of            data: {len(worker_inputs)}')
    print(fr'\_Amount   of     empty  data: {empties}')
    print(fr'\_Amount   of non empty  data: {valids}')
    print(fr'  \_Amount of similar    data: {len(dups)}')
    print(fr'  \_Amount of remaining  data: {len(seen)}')


def multi_process(worker_inputs):
    return _multi(ProcessPool, os.cpu_count() - 1, worker_inputs)


def multi_thread(worker_inputs):
    return _multi(ThreadPool, 4, worker_inputs)
