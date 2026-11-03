"""Exercise the real widget geometry with desktop text scaling enabled."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'
sys.path.insert(0, str(BACKEND))


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Requires an X11 display')
class ScaledLayoutTests(unittest.TestCase):
    def assert_unclipped(self, widget):
        self.assertTrue(widget.winfo_viewable(), str(widget))
        self.assertGreaterEqual(widget.winfo_width(), widget.winfo_reqwidth(), str(widget))
        left, top = widget.winfo_rootx(), widget.winfo_rooty()
        right, bottom = left + widget.winfo_width(), top + widget.winfo_height()
        parent = widget.master
        while parent is not None:
            self.assertGreaterEqual(left, parent.winfo_rootx(), str(widget))
            self.assertGreaterEqual(top, parent.winfo_rooty(), str(widget))
            self.assertLessEqual(right, parent.winfo_rootx() + parent.winfo_width(), str(widget))
            self.assertLessEqual(bottom, parent.winfo_rooty() + parent.winfo_height(), str(widget))
            parent = parent.master

    def test_tabs_and_actions_fit_at_100_150_and_200_percent(self):
        import tkinter as tk
        from tkinter import font as tkfont, ttk
        import posefly_gui
        for scale in (4 / 3, 2., 8 / 3):
            with self.subTest(scale=scale):
                root = tk.Tk(className='Posefly')
                root.withdraw()
                original_scale = root.tk.call('tk', 'scaling')
                errors = []
                root.report_callback_exception = lambda *args: errors.append(args)
                try:
                    root.tk.call('tk', 'scaling', scale)
                    with patch.object(posefly_gui, 'list_available_cameras', return_value=[]):
                        app = posefly_gui.PoseFlyGUI(root)
                    root.deiconify()
                    for size in ('1500x1000', '920x900'):
                        with self.subTest(size=size):
                            root.geometry(size)
                            root.update()
                            self.assertGreaterEqual(app.settings_tabs.winfo_width(), app.settings_tabs.winfo_reqwidth())
                            for button in (*app._toolbar_buttons, app.btn_about):
                                self.assert_unclipped(button)
                            for index in range(3):
                                app.settings_tabs.select(index)
                                root.update()
                                page, scrollbar, actions = app._settings_layouts[index]
                                self.assertGreaterEqual(page.winfo_width(), page.winfo_reqwidth())
                                if index == 0:
                                    buttons = (app.btn_refresh, app.btn_apply_rs)
                                elif index == 1:
                                    buttons = app.feature_switches
                                else:
                                    buttons = (app.save_check, app.btn_browse)
                                for button in buttons:
                                    self.assert_unclipped(button)
                            font = tkfont.Font(root=root, font=ttk.Style(root).lookup('Treeview', 'font'))
                            self.assertGreaterEqual(int(ttk.Style(root).lookup('Treeview', 'rowheight')), font.metrics('linespace') + 8)
                finally:
                    for callback in root.tk.call('after', 'info'):
                        root.after_cancel(callback)
                    root.tk.call('tk', 'scaling', original_scale)
                    root.destroy()
                self.assertEqual(errors, [])
