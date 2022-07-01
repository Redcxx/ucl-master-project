def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h:{minutes}m:{seconds}s'


def get_center_text(text, width, fill_char='='):
    return fill_char * (width // 2 - (len(text) // 2)) + text + fill_char * ((width + 1) // 2 - (len(text) + 1) // 2)
