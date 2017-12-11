import os
import sys
import yaml
from logging.config import dictConfig


def config_logging_yaml(config_file='logging.yaml'):
    ''' Configure the logging module using a YAML file. '''
    config_file = os.path.abspath(config_file)
    try:
        with open(config_file) as f:
            dictConfig(yaml.load(f))
    except IOError:
        print('Error: Can\'t find the configuration file of logging "%s".' % config_file)
        sys.exit(1)
