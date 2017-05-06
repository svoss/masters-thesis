def get_config():
    import os
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini'))

    return config

def load_ex_controller():
    config = get_config()
    import sys,os
    sys.path.append(config.get('controller','loc'))