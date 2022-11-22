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


def parseMongoConfig(file: Union[Path, str], adapt: str = 'MongoObserver') -> Mapping[str, Any]:
  """Parse arguments for the MongoObserver

  INI keys
  [sacred_db]
  host = 
  db_name = 
  user = 
  password = 
  auth_db = 
  auth_mechanism = 

  Args:
      file (Union[Path, str]): the path or str pointing to the ini config
      adapt (str): The output mode, either MongoObserver or IncenseExperimentLoader

  Raises:
      FileNotFoundError: When the path is invalid (also creates a dummy config)
      RuntimeError: If the config file misses some entries

  Returns:
      Mapping[str, Any]: dict of desired arguments
  """
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

      if adapt == 'MongoObserver':
        # adapt to MongoObserver args
        return {
            'url': config['host'],
            'db_name': config['db_name'],
            'username': config['user'],
            'password': config['password'],
            'authSource': config['auth_db'],
            'authMechanism': config['auth_mechanism']
        }
      elif adapt == 'IncenseExperimentLoader':
        # adapt to IncenseExperimentLoader args
        return {
            'mongo_uri':
                f'mongodb://{config["user"]}:{config["password"]}@{config["host"]}/{config["db_name"]}?authMechanism={config["auth_mechanism"]}&authSource={config["auth_db"]}',
            'db_name':
                config['db_name'],
        }
      else:
        raise ValueError(f'Invalid mode "{adapt}"')
