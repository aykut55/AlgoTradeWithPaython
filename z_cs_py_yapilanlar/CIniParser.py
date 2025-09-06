import configparser
import os

class CIniParser:
    def __init__(self, ini_path):
        if not os.path.exists(ini_path):
            raise FileNotFoundError(f"Unable to locate {ini_path}")
            
        self.ini_file_path = ini_path
        self.parser = configparser.ConfigParser()
        # C# versiyonu büyük/küçük harfe duyarsız olduğu için uyumluluk
        self.parser.optionxform = str.upper
        self.parser.read(ini_path)

    def get_setting(self, section_name, setting_name):
        try:
            return self.parser.get(section_name.upper(), setting_name.upper())
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None

    def enum_section(self, section_name):
        if self.parser.has_section(section_name.upper()):
            return self.parser.options(section_name.upper())
        return []

    def add_setting(self, section_name, setting_name, setting_value):
        section = section_name.upper()
        if not self.parser.has_section(section):
            self.parser.add_section(section)
        self.parser.set(section, setting_name.upper(), str(setting_value) if setting_value is not None else None)

    def delete_setting(self, section_name, setting_name):
        section = section_name.upper()
        if self.parser.has_option(section, setting_name.upper()):
            self.parser.remove_option(section, setting_name.upper())

    def save_settings(self, new_file_path=None):
        path_to_save = new_file_path if new_file_path is not None else self.ini_file_path
        with open(path_to_save, 'w') as configfile:
            self.parser.write(configfile)
