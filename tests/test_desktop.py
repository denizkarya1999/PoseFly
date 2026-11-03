"""GNU/Linux desktop registration and window identity regressions."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'))
spec = importlib.util.spec_from_file_location('install_desktop', PROJECT / 'linux/install_desktop.py')
desktop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desktop)


class DesktopTests(unittest.TestCase):
    def test_install_registers_matching_class_and_icon_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            entry = desktop.install(data_home=data)
            content = entry.read_text()
            self.assertIn('StartupWMClass=Posefly\n', content)
            self.assertIn('Icon=posefly\n', content)
            self.assertIn('Exec="' + str(PROJECT / 'PoseFly') + '"\n', content)
            icon = data / 'icons/hicolor/32x32/apps/posefly.png'
            self.assertEqual(icon.read_bytes(), (PROJECT / 'PoseFly Desktop App/PoseFly.WinForms/assets/PoseFly.png').read_bytes())
            modified = entry.stat().st_mtime_ns
            desktop.install(data_home=data)
            self.assertEqual(entry.stat().st_mtime_ns, modified)
            if shutil.which('desktop-file-validate'):
                subprocess.run(['desktop-file-validate', str(entry)], check=True, capture_output=True)

    def test_exec_quotes_special_characters_without_shell_expansion(self):
        self.assertEqual(desktop.desktop_exec(Path('/tmp/my project/100%/PoseFly')), '"/tmp/my project/100%%/PoseFly"')
        self.assertEqual(desktop.desktop_exec(Path('/tmp/"$`/PoseFly')), '"/tmp/\\\\"\\\\$\\\\`/PoseFly"')

    @unittest.skipUnless(os.environ.get('DISPLAY') and shutil.which('xprop'), 'Requires an X11 display and xprop')
    def test_window_exposes_posefly_identity_and_icon(self):
        import tkinter as tk
        import posefly_gui
        root = tk.Tk(className='Posefly')
        try:
            with patch.object(posefly_gui, 'list_available_cameras', return_value=[]):
                app = posefly_gui.PoseFlyGUI(root)
            root.update()
            # With no window manager, Tk's frame() may return its inner window;
            # the named top-level wrapper owns the properties used by taskbars.
            result = subprocess.run(['xprop', '-name', 'PoseFly', 'WM_CLASS'], check=True, capture_output=True, text=True)
            self.assertIn('"Posefly"', result.stdout)
            icon = subprocess.run(['xprop', '-name', 'PoseFly', '-f', '_NET_WM_ICON', '32c', ' $0, $1\\n', '_NET_WM_ICON'],
                                  check=True, capture_output=True, text=True)
            self.assertIn('32, 32', icon.stdout)
        finally:
            for callback in root.tk.call('after', 'info'):
                root.after_cancel(callback)
            root.destroy()
