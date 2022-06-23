import time


def format_time(seconds):
    return time.strftime('%Hh:%Mm:%Ss', time.gmtime(seconds))


def get_center_text(text, width, fill_char='='):
    return fill_char * (width // 2 - (len(text) // 2)) + text + fill_char * ((width + 1) // 2 - (len(text) + 1) // 2)
