# -*- coding:utf-8 -*-

from __future__ import with_statement

import os
import sys
import time

if sys.platform == 'win32':
    import _winreg
    _registry = _winreg.ConnectRegistry(None, _winreg.HKEY_CURRENT_USER)
    def get_runonce():
        return _winreg.OpenKey(_registry,r"Software\Microsoft\Windows\CurrentVersion\Run", 0, _winreg.KEY_ALL_ACCESS)

    def add(name, application):
        key = get_runonce()
        _winreg.SetValueEx(key, name, 0, _winreg.REG_SZ, application)
        _winreg.CloseKey(key)

    def exists(name):
        key = get_runonce()
        exists = True
        try:
            _winreg.QueryValueEx(key, name)
        except WindowsError:
            exists = False
        _winreg.CloseKey(key)
        return exists

    def remove(name):
        key = get_runonce()
        _winreg.DeleteValue(key, name)
        _winreg.CloseKey(key)

else:
    _xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    _xdg_user_autostart = os.path.join(os.path.expanduser(_xdg_config_home), "autostart")

    def getfilename(name):
        return os.path.join(_xdg_user_autostart, name+ ".desktop")

    def add(name, application):
        desktop_entry = "[Desktop Entry]\n"\
            "name=%s\n"\
            "Exec=%s\n"\
            "Type=Application\n"\
            "Terminal=false\n" % (name, application)
        with open(getfilename(name), "w") as f:
            f.write(desktop_entry)

    def exists(name):
        return os.path.exists(getfilename(name))

    def remove(name):
        os.unlink(getfilename(name))

if exists("Friend") == False:
    add("Friend", "python path")
message = "alert Message"
Message = message.encode('utf-8')
print Message
time.sleep(100)
