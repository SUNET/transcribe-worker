import logging
from utils.args import parse_arguments


def get_logger():
    """
    Get a logger instance for the application.
    If the logger already has handlers, it returns the existing logger.
    """

    logger = logging.getLogger(__name__)
    _, _, _, _, debug, logfile = parse_arguments()

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        )

        if logfile != "":
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    if debug is True:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode is enabled.")
    else:
        logger.setLevel(logging.INFO)

    return logger
