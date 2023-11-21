import sys
import os
import logging
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from typing import Optional


def incremental_path(path, select_last=False):
    """
    Create a directory or file with an incremental number if a directory or file with the same name already exists.
    """

    # Define the base path and extension
    base_path, extension = os.path.splitext(path)

    # Check if extension is empty
    is_file = (extension != '')

    # Add an incremental number if a directory or file with the same name already exists
    increment = 1
    while os.path.exists(path):
        path = f'{base_path}_{increment}{extension}'
        increment += 1
    
    if select_last:
        path = f'{base_path}_{increment-2}{extension}'
    else:
        # Create the directory or file
        if is_file:
            with open(path, 'w'):
                pass
        else:
            os.makedirs(path)

    return path

def setup_logging(module_name:str, logger:logging.Logger, logdir:str=None, verbose:str=True):

    # Set the logger's level
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s [%(module)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers = [console_handler]

    if logdir is not None:
        file_handler = logging.FileHandler(os.path.join(logdir, f'{module_name}.log'), mode='w')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Set the handlers for the logger
    logger.handlers = handlers