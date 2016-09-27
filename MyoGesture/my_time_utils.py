import time

def elapsed_time(begin):
    return round(time.monotonic() - begin, 2)

def begin():
    return time.monotonic()
