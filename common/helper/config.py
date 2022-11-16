import ini
from pathlib import Path
from typing import Union, Mapping, Any

DEFAULT_MONGODB_CONFIG = {
    'sacred_db': {
        'host': '<host>',
        'db_name': '<sacred_db_name>',
        'user': '<username>',
        'password': '<password>',
        'auth_db': '<password>',
        'auth_mechanism': 'SCRAM-SHA-1',
    }
}


def parseMongoObserverArgs(file: Union[Path, str]) -> Mapping[str, Any]:
  path = Path(file) if isinstance(file, str) else file

  if not path.exists():
    with path.open('w') as f:
      f.write(ini.encode(DEFAULT_MONGODB_CONFIG))
      raise FileNotFoundError(f'Created dummy config at {path}')
  else:
    with path.open('r') as f:
      config = ini.parse(f.read())['sacred_db']
      if not all([entry in config for entry in DEFAULT_MONGODB_CONFIG['sacred_db']]):
        raise RuntimeError(f'Incomplete config file {path}')

      # adapt to MongoObserver args
      return {
          'url': config['host'],
          'db_name': config['db_name'],
          'username': config['user'],
          'password': config['password'],
          'authSource': config['auth_db'],
          'authMechanism': config['auth_mechanism']
      }
