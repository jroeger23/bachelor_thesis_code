import logging
import sys

# Setup global logging

logging.addLevelName(logging.DEBUG, 'DBG')
logging.addLevelName(logging.INFO, 'INF')
logging.addLevelName(logging.WARNING, 'WRN')
logging.addLevelName(logging.ERROR, 'ERR')
logging.addLevelName(logging.FATAL, 'FTL')

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(levelname)s][%(name)s] %(message)s'))


logging.basicConfig(level=logging.INFO, handlers=[handler])