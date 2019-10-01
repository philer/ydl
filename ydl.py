#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""YDL - An interactive CLI for youtube-dl

Usage:
    ydl [-pc] [-a | --no-archive]
    ydl -h

Options:
    -p --play       Automatically play videos once they have
                    finished downloading
    -c --continue   Resume unfinishe download (existing .part/.ytdl files)
    -a --archive-filename
                    File to be used for storing status and URL of downloads
    --no-archive    Start a completely empty session, do not use an archive file
    -h --help       Show this help message and exit.

test vine (spider):
https://www.youtube.com/watch?v=RwJe8KfPCEQ

TODO:
* tests
* delete option (show/hide)
* track missing videos (deleted from outside)
* video playback:
    + finish order vs. adding order
    + delete option after play
    + play arbitrary (separate of queue)
    + countdown timer for next playback
"""

from __future__ import annotations
from typing import Any, Iterable, Mapping, Optional, Tuple

import os
import sys
import signal
from collections import defaultdict
from itertools import count
from functools import partial
from urllib.parse import urlparse

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from docopt import docopt
from urwid import (AttrMap, Columns, Divider, Edit, ExitMainLoop, Frame,
                   ListBox, MainLoop, Pile, Text, WidgetWrap,
                   AsyncioEventLoop)
import youtube_dl
import pyperclip

VIDEO_PLAYER = "/usr/bin/mpv"
VIDEO_DELAY = 2
ARCHIVE_FILENAME = ".ydl_archive"
MAX_POOL_SIZE = 5
MAX_HOST_POOL_SIZE = 3


class MuteLogger:
    def _noop(self, _: Any):
        pass
    debug = warning = error = _noop

ydl_settings = {
    "format": "best",
    "noplaylist": True,
    "call_home": False,
    "cachedir": False,
    # "download_archive": ".youtube_dl_archive",
    "logger": MuteLogger(),
    "outtmpl": "%(extractor)s-%(id)s_%(title)s.%(ext)s"
    # "outtmpl": "/tmp/ydl/" + youtube_dl.DEFAULT_OUTTMPL,
}


class Archive:
    """Simple database file for remembering past downloads."""

    _video_properties = "status", "extractor", "id", "ext", "url", "title"

    def __init__(self, filename: Optional[str]=None):
        self._filename = filename or ARCHIVE_FILENAME

    def __iter__(self) -> Iterator[Tuple[str, str, str, str, str]]:
        """Iterate over the URLs with status in the archive file."""
        try:
            with open(self._filename, "rt") as archive:
                for line in map(str.rstrip, archive):
                    if line:
                        parts = line.split(None, len(self._video_properties) - 1)
                        yield dict(zip(self._video_properties, parts))
        except FileNotFoundError:
            pass

    def update(self, videos: Iterable[Video]) -> None:
        """
        Rewrite archive file with updated status for known URLs
        and appended new ones.
        """
        # TODO auto-backup
        with open(self._filename, "wt") as archive:
            for video in videos:
                if video.status != "error":
                    parts = (getattr(video, prop) for prop in self._video_properties)
                    archive.write(" ".join(parts) + "\n")


class DownloadManager:
    """Manages video download workers and parallel connections per host."""

    def __init__(self):
        self._executors = defaultdict(partial(ThreadPoolExecutor, max_workers=MAX_HOST_POOL_SIZE))
        self._max_workers_sem = asyncio.BoundedSemaphore(value=MAX_POOL_SIZE)
        self._interrupted = False

    async def download(self, video):
        async with self._max_workers_sem:
            if self._interrupted:
                return
            ex = self._executors[video.hostname]
            await asyncio.get_running_loop().run_in_executor(ex, self._work, video)

    def shutdown(self):
        self._interrupted = True
        for ex in self._executors.values():
            ex.shutdown(wait=True)

    class Interrupt(Exception):
        """Used to break out of youtube-dl's blocking download via hook."""
        pass

    def _raise_interrupt(self, _):
        """Called by youtube-dl periodically - used to break out when we want to quit."""
        if self._interrupted:
            raise self.Interrupt()

    def _work(self, video):
        """Perform actual downloads - this is run in a thread."""
        hooks = [self._raise_interrupt, video.set_download_info]
        ydl = youtube_dl.YoutubeDL(dict(ydl_settings, progress_hooks=hooks))
        video.status = "downloading"
        try:
            ydl.download([video.url])
        except youtube_dl.utils.DownloadError:
            video.status = "error"
        except self.Interrupt:
            pass


class PlaylistManager:
    def __init__(self, delay=VIDEO_DELAY):
        self.delay = delay
        self._videos = asyncio.Queue()
        self._task = None
        self._process = None

    async def add_video(self, video):
        await self._videos.put(video)
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def shutdown(self):
        if self._task:
            self._task.cancel()
        if self._process:
            self._process.terminate()

    async def _run(self):
        while True:
            video = await self._videos.get()
            with open(os.devnull, 'w') as devnull:
                proc_task = asyncio.create_subprocess_exec(VIDEO_PLAYER,
                                                           video.filename,
                                                           stdout=devnull,
                                                           stderr=devnull)
                self._process = await asyncio.shield(proc_task)
                await self._process.wait()
                self._process = None
            self._videos.task_done()
            await asyncio.sleep(self.delay)


class Video(WidgetWrap):
    """Ugly mix of data model and view widget"""
    status_icon = {
        "pending": " ⧗ ",
        "downloading": " ⬇ ",
        "finished": " ✔ ",
        "error": " ⨯ ",
    }

    @property
    def hostname(self):
        return urlparse(self.url).hostname

    @property
    def filename(self):
        return f"{self.extractor}-{self.id}_{self.title}.{self.ext}"

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status
        with self._ui.draw_lock:
            self._root.set_attr_map({None: status})
            if status == "downloading" and self.progress:
                self._status_widget.set_text(f"{self.progress: >3.0%}")
            else:
                self._status_widget.set_text(self.status_icon[status])
            self._ui._loop.draw_screen()

    def __init__(self, ui, url, status="pending", **meta):
        self._ui = ui
        self.url = url
        self._status = status
        self._assign_meta(meta)
        self.progress = 0

        self._status_widget = Text(self.status_icon[status])
        try:
            info = f"{self.id} - {self.title}"
        except AttributeError:
            info = url
        self._info_widget = Text(info, wrap='clip')
        columns = [
            (3, self._status_widget),
            (1, Text(("divider", "│"))),
            self._info_widget,
        ]
        self._root = AttrMap(Columns(columns, dividechars=1), status)
        super().__init__(self._root)

    def _assign_meta(self, data):
        for prop in "extractor", "id", "title", "ext":
            try:
                setattr(self, prop, data[prop])
            except KeyError:
                pass

    def prepare_meta(self):
        """
        Download meta data for this video. Some of the classes functionality
        will not be available until this is done.
        """
        ydl = youtube_dl.YoutubeDL(ydl_settings)
        try:
            meta = ydl.extract_info(self.url, download=False)
        except youtube_dl.utils.DownloadError:
            self.status = "error"
        else:
            self._assign_meta(meta)
            with self._ui.draw_lock:
                self._info_widget.set_text(f"{self.id} - {self.title}")
                self._ui._loop.draw_screen()

    def set_download_info(self, data):
        # from https://github.com/ytdl-org/youtube-dl/blob/master/youtube_dl/YoutubeDL.py
        # * status: One of "downloading", "error", or "finished".
        #           Check this first and ignore unknown values.
        # If status is one of "downloading", or "finished", the
        # following properties may also be present:
        # * filename: The final filename (always present)
        # * tmpfilename: The filename we're currently writing to
        # * downloaded_bytes: Bytes on disk
        # * total_bytes: Size of the whole file, None if unknown
        # * total_bytes_estimate: Guess of the eventual file size,
        #                         None if unavailable.
        # * elapsed: The number of seconds since download started.
        # * eta: The estimated time in seconds, None if unknown
        # * speed: The download speed in bytes/second, None if
        #          unknown
        # * fragment_index: The counter of the currently
        #                   downloaded video fragment.
        # * fragment_count: The number of fragments (= individual
        #                   files that will be merged)
        if data["status"] == "downloading":
            try:
                total_bytes = data["total_bytes"]
            except KeyError:
                total_bytes = data.get("total_bytes_estimate")
            if total_bytes:
                self.progress = data["downloaded_bytes"] / total_bytes
        else:
            self.progress = 0
        self.status = data["status"]  # triggers redraw

    def render(self, size, focus=False):
        """hack allows me to change text background based on size"""
        info = self._info_widget.text.strip().ljust(size[0])
        filled_width = int(size[0] * self.progress)
        filled, empty = info[:filled_width], info[filled_width:]
        self._info_widget.set_text([("progress_filled", filled), empty])
        return self._root.render(size, focus)


class CustomEventLoop(AsyncioEventLoop):
    """
    Urwid's AsyncioEventLoop's exception handling is broken.
    This code is from https://github.com/urwid/urwid/issues/235#issuecomment-458483802
    """
    def _exception_handler(self, loop, context):
        exc = context.get('exception')
        if exc:
            loop.stop()
            if not isinstance(exc, ExitMainLoop):
                # Store the exc_info so we can re-raise after the loop stops
                import sys
                self._exc_info = sys.exc_info()
                if self._exc_info == (None, None, None):
                    self._exc_info = (type(exc), exc, exc.__traceback__)
        else:
            loop.default_exception_handler(context)

class Ui:
    palette = [
        ("pending",         "",            "", "", "g70",  ""),
        ("downloading",     "light blue",  "", "", "#6dd", ""),
        ("finished",        "light green", "", "", "#8f6", ""),
        ("error",           "light red",   "", "", "#d66", ""),
        ("progress_filled", "standout",    "", "", "g0",   "#6dd"),
        ("divider",         "",            "", "", "#666", ""),
        ("prompt",          "light green", ""),
    ]

    def __init__(self, core, aio_loop):
        self._core = core
        self._aio_loop = aio_loop

        self._input = Edit(caption=("prompt", "⟩ "))
        self._videos = ListBox([])
        footer = Pile([AttrMap(Divider("─"), "divider"), self._input])
        self._root = Frame(body=self._videos, footer=footer)
        self._root.focus_position = "footer"

        self._loop = MainLoop(widget=self._root,
                              palette=self.palette,
                              event_loop=CustomEventLoop(loop=aio_loop),
                              unhandled_input=self._handle_global_input)
        self._loop.screen.set_terminal_properties(
            colors=256, bright_is_bold=False, has_underline=True)

        self.draw_lock = threading.RLock()

    def run_loop(self):
        self._loop.run()

    def halt_loop(self, *_):
        raise ExitMainLoop()

    def add_video(self, video):
        with self.draw_lock:
            self._videos.body.append(video)

    def _handle_global_input(self, key):
        if key == 'esc':
            self._core.shutdown()
        elif key == 'ctrl v':
            try:
                self._handle_urls(pyperclip.paste())
            except pyperclip.PyperclipException:
                pass
        elif key == 'enter' and self._input.edit_text:
            self._handle_urls(self._input.edit_text)
            self._input.edit_text = ""

    def _handle_urls(self, text):
        """Extract valid urls from user input and pass them on."""
        for url in text.split():
            if url:
                video = self._core.create_video(url)
                self._aio_loop.create_task(self._core.queue_video(video))

class YDL:
    """Core controller"""
    def __init__(self, play=False, resume=False, use_archive=True, archive_filename=None):
        self._aio_loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self.ui = Ui(self, self._aio_loop)

        self.downloads = DownloadManager()
        self.playlist = PlaylistManager() if play else None

        self.archive = Archive(archive_filename) if use_archive else None
        self.videos = dict()
        if self.archive:
            for data in self.archive:
                video = self.create_video(**data)
                if video.status in {"pending", "downloading"}:
                    self._aio_loop.create_task(self.queue_video(video))

    def run(self):
        self._aio_loop.add_signal_handler(signal.SIGINT, self.shutdown)
        self.ui.run_loop()
        self._aio_loop.remove_signal_handler(signal.SIGINT)
        self.archive.update(self.videos.values())

    def shutdown(self):
        if self.playlist:
            self.playlist.shutdown()
        self.downloads.shutdown()
        self.ui.halt_loop()

    def create_video(self, url, status="pending", **meta):
        try:
            video = self.videos[url]
        except KeyError:
            video = Video(self.ui, url, status, **meta)
            self.videos[video.url] = video
        self.ui.add_video(video)
        return video

    async def queue_video(self, video):
        if video.status == "pending":
            await self._aio_loop.run_in_executor(self._executor, video.prepare_meta)
        if video.status == "error":
            return
        if video.status != "finished":
            await self.downloads.download(video)
        if self.playlist:
            await self.playlist.add_video(video)


if __name__ == "__main__":
    args = docopt(__doc__, version="0.0.1")
    ydl = YDL(play=args["--play"],
              resume=args["--continue"],
              use_archive=not args["--no-archive"],
              archive_filename=args["--archive-filename"])
    ydl.run()
