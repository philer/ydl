#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""YDL - An interactive CLI for youtube-dl

Usage:
    ydl [-pc] [-a | --no-archive] [-v|-vv]
    ydl -h

Options:
    -p --play       Automatically play videos once they have
                    finished downloading
    -a --archive-filename
                    File to be used for storing status and URL of downloads
    --no-archive    Start a completely empty session, do not use an archive file
    -v --verbose   Show more and more info.
    -h --help       Show this help message and exit.
"""

import io
import logging

from docopt import docopt

from . import __version__, Archive, YDL


def setup_logging(verbosity: int):
    root_logger = logging.getLogger(__package__)
    root_logger.setLevel((logging.ERROR, logging.INFO, logging.DEBUG)[verbosity])
    log_stream = io.StringIO()
    if verbosity > 1:
        fmt = "%(levelname)s (%(name)s) %(message)s"
    else:
        fmt = "%(message)s"
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(handler)
    return log_stream

def main():
    args = docopt(__doc__, version=__version__)
    log_stream = setup_logging(args["--verbose"])
    archive = None if args["--no-archive"] else Archive(args["--archive-filename"])
    ydl = YDL(archive, play=args["--play"])
    try:
        ydl.run()
    finally:
        print(log_stream.getvalue(), end="")

if __name__ == "__main__":
    main()
