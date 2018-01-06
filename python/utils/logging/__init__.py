import os
import sys
import yaml
import logging.config


def config_logging_yaml(config_file='logging.yaml'):
    ''' Configure the logging module using a YAML file. '''
    # Change the default text of level name.
    logging.addLevelName(logging.DEBUG, ' DEBUG ')
    logging.addLevelName(logging.INFO, ' INFOR ')
    logging.addLevelName(logging.ERROR, ' ERROR ')

    config_file = os.path.abspath(config_file)
    try:
        with open(config_file) as f:
            logging.config.dictConfig(yaml.load(f))
    except IOError:
        print('Error: Can\'t find the configuration file of logging "%s".' % config_file)
        sys.exit(1)
