from common.logger import LoggerManager

LoggerManager()

logger = LoggerManager.get_logger("DatasetVerifier")

logger.info("Dataset verification started.")

logger.warning("This is a warning.")

logger.error("This is an error.")