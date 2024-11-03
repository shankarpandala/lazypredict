import logging

class Logger:
    """
    Logger configuration for LazyPredict.

    Methods
    -------
    configure_logger(name, level=logging.INFO):
        Configures and returns a logger.
    """

    @staticmethod
    def configure_logger(name, level=logging.INFO):
        """
        Configure and return a logger.

        Parameters
        ----------
        name : str
            Name of the logger.
        level : int, optional
            Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.

        Returns
        -------
        Logger
            Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
