import yaml
import os


class Config:
    def __init__(self, yaml_path=None, **kwargs):
        self.config_file_name = "config.yaml"
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        if yaml_path is None:
            yaml_path = os.path.join(self.current_file_dir, self.config_file_name)

        config_data = self.load(yaml_path)
        self.__dict__ = config_data
        self.__dict__.update(kwargs)

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.current_file_dir, self.config_file_name)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data
