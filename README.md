# YDL

An interactive Youtube-DL wrapper for the command line with playback option.

## Usage
        ydl [-pc] [-a | --no-archive] [-v|-vv]
        ydl -h

## Options
        -p --play       Automatically play videos once they have
                        finished downloading
        -c --continue   Resume unfinishe download (existing .part/.ytdl files)
        -a --archive-filename
                        File to be used for storing status and URL of downloads
        --no-archive    Start a completely empty session, do not use an archive file
        -v --verbose   Show more and more info.
        -h --help       Show this help message and exit.
(For an up-to-date version of usage instructions see ydl/__main__.py)

## Dependencies

* youtube_dl
* urwid
* docopt
* pyperclip (for pasting URLs)
