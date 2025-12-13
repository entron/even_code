import logging
import os


def setup_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a file handler for logging
    log_directory = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, f"{name}.log")
    file_handler = logging.FileHandler(log_file)

    # Create a console handler for logging
    console_handler = logging.StreamHandler()

    # Set the logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
