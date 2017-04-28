def get_config():
    import os
    import ConfigParser
    config = ConfigParser.RawConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini'))

    return config