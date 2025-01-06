#!/usr/bin/env python3

import click
import pathlib
import shutil
import subprocess
import re
import os


@click.command()
def main():
    repo_dir = pathlib.Path(__file__).parent.parent
    nb_dir = repo_dir / 'notebooks'
    scripts_dir = repo_dir / 'scripts'
    src_dir = repo_dir / 'src'
    subprocess.run(['python', src_dir / 'workshop_utils' / 'fetch.py'], cwd=repo_dir)
    docs_nb_dir = repo_dir / 'docs' / 'source' / 'full'
    for f in docs_nb_dir.glob('**/*md'):
        output_f = (nb_dir / f.parent.name / f.name.replace('md', 'ipynb')).absolute()
        output_f.parent.mkdir(exist_ok=True)
        subprocess.run(['jupytext', f.absolute(), '-o', output_f,
                        '--from', 'myst'], cwd=repo_dir)
        nb_contents = re.sub(r'../../_static/', r'../../docs/source/_static/',
                             output_f.read_text())
        output_f.write_text(nb_contents)


if __name__ == '__main__':
    main()
