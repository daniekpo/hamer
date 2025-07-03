import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_output(stdout=True, stderr=True):
    devnull = os.open(os.devnull, os.O_WRONLY)

    # Store original file descriptors
    old_fds = {}
    if stdout:
        old_fds['stdout'] = (sys.stdout.fileno(), os.dup(sys.stdout.fileno()))
    if stderr:
        old_fds['stderr'] = (sys.stderr.fileno(), os.dup(sys.stderr.fileno()))

    try:
        if stdout:
            os.dup2(devnull, sys.stdout.fileno())
        if stderr:
            os.dup2(devnull, sys.stderr.fileno())
        yield
    finally:
        # Restore original fds
        for name, (fd, saved_fd) in old_fds.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(devnull)