#!/usr/bin/env python3
"""Register this checkout in the current user's GNU/Linux application menu."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parent.parent


def desktop_string(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def desktop_exec(path):
    # Desktop-entry string escaping is applied after Exec argument quoting.
    value = str(path).replace('%', '%%')
    for character in ('\\', '"', '`', '$'):
        value = value.replace(character, '\\' + character)
    return desktop_string('"' + value + '"')


def desktop_entry(project):
    return '\n'.join([
        '[Desktop Entry]', 'Version=1.0', 'Type=Application', 'Name=PoseFly',
        'Comment=Drone detection and optical sensing',
        'Exec=' + desktop_exec(project / 'PoseFly'),
        'Icon=posefly', 'Terminal=false', 'Categories=Science;',
        'StartupNotify=false', 'StartupWMClass=Posefly', '',
    ])


def install(project=PROJECT, data_home=None):
    data_home = Path(data_home or os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    applications = data_home / 'applications'
    icons = data_home / 'icons/hicolor/32x32/apps'
    applications.mkdir(parents=True, exist_ok=True)
    icons.mkdir(parents=True, exist_ok=True)
    icon = icons / 'posefly.png'
    source = project / 'PoseFly Desktop App/PoseFly.WinForms/assets/PoseFly.png'
    if not icon.exists() or icon.read_bytes() != source.read_bytes():
        shutil.copyfile(source, icon)
    entry = applications / 'posefly.desktop'
    content = desktop_entry(project)
    if not entry.exists() or entry.read_text() != content:
        entry.write_text(content)
        updater = shutil.which('update-desktop-database')
        if updater:
            subprocess.run([updater, str(applications)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return entry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    try:
        entry = install()
        if not args.quiet:
            print(f'Installed {entry}')
    except OSError as exc:
        parser.exit(1, f'PoseFly desktop integration: {exc}\n')
