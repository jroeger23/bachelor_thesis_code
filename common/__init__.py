import logging
import sys

# Setup global logging

logging.addLevelName(logging.DEBUG, 'DBG')
logging.addLevelName(logging.INFO, 'INF')
logging.addLevelName(logging.WARNING, 'WRN')
logging.addLevelName(logging.ERROR, 'ERR')
logging.addLevelName(logging.FATAL, 'FTL')


class MultilineHandler(logging.StreamHandler):

  def __init__(self):
    super(MultilineHandler, self).__init__()

  def emit(self, record):
    lines = record.msg.split('\n')
    for l in lines:
      record.msg = l
      super(MultilineHandler, self).emit(record)


handler = MultilineHandler()
handler.setFormatter(logging.Formatter('[%(levelname)s][%(name)s] %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[handler])