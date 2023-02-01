import sys

from loguru import logger

# try:
#     logger.remove(0)
# except ValueError:
#     pass
#
# logger.add(
#     # colorize=False,
#     # format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
#     format='{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - <level>{message}</level>',
#     level='INFO',
#     enqueue=True,
#     sink=sys.stdout,
# )
logger.info('Import BenchmarkToolsOptimizers')
