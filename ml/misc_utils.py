import time


def format_time(seconds, datetime=False):
    if datetime:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h:{int(minutes)}m:{int(seconds)}s'


def get_center_text(text, width, fill_char='='):
    return fill_char * (width // 2 - (len(text) // 2)) + text + fill_char * ((width + 1) // 2 - (len(text) + 1) // 2)
