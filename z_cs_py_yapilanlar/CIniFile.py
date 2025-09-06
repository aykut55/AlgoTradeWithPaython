import os
import sys
import ctypes
from ctypes import wintypes

class CIniFile:
    def __init__(self, ini_file_name=None):
        self.path = None
        self.exe_name = os.path.splitext(os.path.basename(sys.executable))[0]

        self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        self.GetPrivateProfileString = self.kernel32.GetPrivateProfileStringW
        self.GetPrivateProfileString.argtypes = [
            wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
            wintypes.LPWSTR, wintypes.DWORD, wintypes.LPCWSTR]
        self.GetPrivateProfileString.restype = wintypes.DWORD

        self.WritePrivateProfileString = self.kernel32.WritePrivateProfileStringW
        self.WritePrivateProfileString.argtypes = [
            wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR]
        self.WritePrivateProfileString.restype = wintypes.BOOL

        if ini_file_name:
            self.set_file_name(ini_file_name)

    def set_file_name(self, name):
        self.path = os.path.abspath(name)

    def read(self, key, section=None):
        if self.path is None:
            return ""
        
        section_to_use = section if section is not None else self.exe_name
        buffer = ctypes.create_unicode_buffer(255)
        self.GetPrivateProfileString(section_to_use, key, "", buffer, len(buffer), self.path)
        return buffer.value

    def write(self, key, value, section=None):
        if self.path is None:
            return
            
        section_to_use = section if section is not None else self.exe_name
        self.WritePrivateProfileString(section_to_use, key, value, self.path)

    def delete_key(self, key, section=None):
        self.write(key, None, section)

    def delete_section(self, section=None):
        self.write(None, None, section)

    def key_exists(self, key, section=None):
        return len(self.read(key, section)) > 0
