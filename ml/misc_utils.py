import time


def format_time(seconds):
    return time.strftime('%Hh:%Mm:%Ss', time.gmtime(seconds))
