import json
from types import SimpleNamespace
CONFIG_FILE_PATH = "config.json"

class ConfigurationProvider(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigurationProvider, cls).__new__(cls)
            return cls.instance
        
    def __init__(self):
        self._config = object()
        with open(CONFIG_FILE_PATH, "r") as configFile:
            self._config = json.load(
                configFile,
                object_hook = lambda hook: SimpleNamespace(**hook))
        configFile.close()
    
    def GetConfiguration(self, sectionName):
        return self._config.__getattribute__(sectionName)

__CONFIGURATION_PROVIDER = ConfigurationProvider()
configurationProvider = __CONFIGURATION_PROVIDER