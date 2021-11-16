import os
import logging
from logging import handlers


class LoggingWrapper:
    """ """
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    def __init__(self, name):
        self.m_logger = logging.getLogger(name)

        self.basic_level = logging.INFO
        self.basic_formatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        self.basic_file_max_bytes = 10 * 1024 * 1024  # 10MB
        self.basic_file_max_count = 10

        self.m_logger.setLevel(logging.DEBUG)

    def add_file_handler(self, path, level=None, formatter=None):
        """ Add new file handler to logger. """
        if level == None:
            level = self.basic_level

        if formatter == None:
            formatter = self.basic_formatter

        dirpath = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=path,
            maxBytes=self.basic_file_max_bytes,
            backupCount=self.basic_file_max_count)
        file_handler.setLevel(level)
        formatter = logging.Formatter(formatter)
        file_handler.setFormatter(formatter)

        self.m_logger.addHandler(file_handler)

    def add_stream_handler(self, stream=None, level=None, formatter=None):
        """ Add new stream handler to logger
            Parameter
              stream : stream IO of handler. If None(Default), it use stderr
        """
        if level == None:
            level = self.basic_level

        if formatter == None:
            formatter = self.basic_formatter

        stream_handler = logging.StreamHandler(stream)

        stream_handler.setLevel(level)
        formatter = logging.Formatter(formatter)
        stream_handler.setFormatter(formatter)

        self.m_logger.addHandler(stream_handler)

    def _log(self, msg, level):
        _logger = self.m_logger

        if level == logging.CRITICAL:
            _logger.critical(msg)
        elif level == logging.ERROR:
            _logger.error(msg)
        elif level == logging.WARNING:
            _logger.warning(msg)
        elif level == logging.INFO:
            _logger.info(msg)
        elif level == logging.DEBUG:
            _logger.debug(msg)
        elif level == logging.NOTSET:
            _logger.info(msg)
        else:
            pass

    def critical(self, msg):
        self._log(msg, logging.CRITICAL)

    def error(self, msg):
        self._log(msg, logging.ERROR)

    def warning(self, msg):
        self._log(msg, logging.WARNING)

    def info(self, msg):
        self._log(msg, logging.INFO)

    def debug(self, msg):
        self._log(msg, logging.DEBUG)


if __name__ == "__main__":
    logger = LoggingWrapper(__name__)
    logger.add_stream_handler(None, logging.DEBUG, None)
    logger.add_file_handler('./log/log-test', logging.DEBUG, None)

    for i in range(1):
        logger.critical("Critical log")
        logger.error("Error log")
        logger.warning("Warning log")
        logger.info("Info log")
        logger.debug("Debug log")
