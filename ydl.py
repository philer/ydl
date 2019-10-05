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
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple

import os
import sys
import shutil
import signal
from collections import defaultdict
import dataclasses
from itertools import count
from functools import partial
from urllib.parse import urlparse
from datetime import date

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
VIDEO_DELAY = 0
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


@dataclasses.dataclass
class Video:
    """Primary model."""
    url: str
    status: str = "pending"

    _meta_properties = "extractor", "id", "title", "ext"
    extractor: str = None
    id: str = None
    title: str = None
    ext: str = None

    def __post_init__(self):
        self.progress = 0
        self.observers = []
        self.finished = asyncio.get_event_loop().create_future()

    def __setattr__(self, name, value):
        """A simplistic observer pattern."""
        super().__setattr__(name, value)
        try:
            for observer in self.observers:
                observer(self, name, value)
        except AttributeError:  # missing self.observers during __init__
            pass

    @property
    def hostname(self):
        return urlparse(self.url).hostname

    @property
    def filename(self):
        if None in (self.extractor, self.id, self.title, self.ext):
            raise AttributeError(f"Can't generate filename for {self}.")
        return f"{self.extractor}-{self.id}_{self.title}.{self.ext}"

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
            for prop in self._meta_properties:
                try:
                    setattr(self, prop, meta[prop])
                except KeyError:
                    pass

    def sync_to_original(self, original):
        """
        This video has been identified as a duplicate
        and should have its meta info mirror that of the original.
        """
        self.status = "duplicate"
        for prop in self._meta_properties:
            setattr(self, prop, getattr(original, prop))
        original.observers.append(self._handle_update)

    def _handle_update(self, original, prop, value):
        """Update a meta property mirrored from another instance."""
        if prop in self._meta_properties:
            setattr(self, prop, value)

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
        self.status = data["status"]
        if data["status"] == "downloading":
            try:
                total_bytes = data["total_bytes"]
            except KeyError:
                total_bytes = data.get("total_bytes_estimate")
            if total_bytes:
                self.progress = data["downloaded_bytes"] / total_bytes
        elif data["status"] == "finished":
            self.finished.set_result(True)


class Archive:
    """Simple database file for remembering past downloads."""

    """
    Properties stored in archive.
    Only status and url exist reliably, so they are first.
    """
    _video_properties = "status", "url", "extractor", "id", "ext", "title"

    def __init__(self, filename: Optional[str]=None):
        self._filename = filename or ARCHIVE_FILENAME

    def __iter__(self) -> Iterator[Video]:
        """Iterate over the URLs with status in the archive file."""
        try:
            with open(self._filename, "rt") as archive:
                for lineno, line in enumerate(map(str.rstrip, archive)):
                    if line:
                        parts = line.split(None, len(self._video_properties) - 1)
                        if len(parts) in {2, 6}:
                            yield Video(**dict(zip(self._video_properties, parts)))
                        else:
                            raise ValueError(f"Invalid archive state on line {lineno}: '{line}'")
        except FileNotFoundError:
            pass

    def update(self, videos: Iterable[Video]) -> None:
        """
        Rewrite archive file with updated status for known URLs
        and appended new ones.
        """

        # one backup per day keeps sorrow at bay
        if os.path.isfile(self._filename):
            backup = f"{self._filename}.{date.today():%Y-%m-%d}.backup"
            if not os.path.isfile(backup):
                shutil.copyfile(self._filename, backup)

        with open(self._filename, "wt") as archive:
            for video in videos:
                if video.status not in {"error", "duplicate"}:
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
        self._playlist = asyncio.Queue()
        self._task = None
        self._process = None

    def add_video(self, video):
        asyncio.create_task(self._playlist.put(video.filename))
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def shutdown(self):
        if self._task:
            self._task.cancel()
        if self._process:
            self._process.terminate()
        if self._task:
            try:
                exc = self._task.exception()
            except asyncio.exceptions.InvalidStateError:
                pass
            else:
                if not isinstance(exc, asyncio.CancelledError):
                    raise exc

    async def _run(self):
        while True:
            filename = await self._playlist.get()
            with open(os.devnull, 'w') as devnull:
                proc_task = asyncio.create_subprocess_exec(VIDEO_PLAYER,
                                                           filename,
                                                           stdout=devnull,
                                                           stderr=devnull)
                self._process = await asyncio.shield(proc_task)
                await self._process.wait()
                self._process = None
            self._playlist.task_done()
            await asyncio.sleep(self.delay)


class VideoWidget(WidgetWrap):
    """Ugly mix of data model and view widget"""
    status_icon = {
        "pending": " ⧗ ",
        "duplicate": " = ",
        "downloading": " ⬇ ",
        "finished": " ✔ ",
        "error": " ⨯ ",
    }

    @property
    def _info(self):
        if self._video.id is not None and self._video.title is not None:
            return f"{self._video.id} - {self._video.title}"
        return self._video.url

    def __init__(self, ui, video):
        self._ui = ui
        self._video = video

        self._status_widget = Text(self.status_icon[video.status])
        self._info_widget = Text(self._info, wrap='clip')
        self._divider = Text(("divider", "│"))
        columns = [
            (3, self._status_widget),
            (1, self._divider),
            self._info_widget,
        ]
        self._root = AttrMap(Columns(columns, dividechars=1), video.status, video.status + "_focus")
        super().__init__(self._root)

        video.observers.append(self._handle_update)

    def _handle_update(self, _video, prop, _value):
        if prop in {"id", "title"}:
            self.update_info()
        elif prop in {"status", "progress"}:
            self.update_status()

    def update_info(self):
        with self._ui.draw_lock:
            self._info_widget.set_text(self._info)
            self._ui._loop.draw_screen()

    def update_status(self):
        if self._video.status == "downloading" and self._video.progress:
            icon = f"{self._video.progress: >3.0%}"
        else:
            icon = self.status_icon[self._video.status]
        with self._ui.draw_lock:
            self._root.set_attr_map({None: self._video.status})
            self._root.set_focus_map({None: self._video.status + "_focus"})
            self._status_widget.set_text(icon)
            self._ui._loop.draw_screen()

    def render(self, size, focus=False):
        """hack allows me to change text background based on size"""
        info = self._info.ljust(size[0])
        if self._video.status == "downloading":
            filled_width = int(size[0] * self._video.progress)
            filled, empty = info[:filled_width], info[filled_width:]
            self._info_widget.set_text([("progress_filled", filled), empty])
        else:
            self._info_widget.set_text(info)
        self._divider.set_text("┃" if focus else "│")  # ╽╿
        self._invalidate()
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
        ("duplicate",       "yellow",      "", "", "#da0", ""),
        ("downloading",     "light blue",  "", "", "#6dd", ""),
        ("finished",        "light green", "", "", "#8f6", ""),
        ("error",           "light red",   "", "", "#d66", ""),
        ("progress_filled", "standout",    "", "", "g0",   "#6dd"),
        ("divider",         "",            "", "", "#666", ""),
        ("prompt",          "light green", ""),
    ]
    for display_attribute in palette[:5]:
        palette.append((display_attribute[0] + "_focus",
                        display_attribute[1] + ",bold",
                        *display_attribute[2:4],
                        display_attribute[4] + ",bold",
                        display_attribute[5]))


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
                self._aio_loop.create_task(self._core.add_video(Video(url)))

    def add_video(self, video):
        """Wrap a new video and add it to the display."""
        widget = VideoWidget(self, video)
        with self.draw_lock:
            self._videos.body.append(widget)


class YDL:
    """Core controller"""
    def __init__(self, play=False, resume=False, use_archive=True, archive_filename=None):
        self._aio_loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self.ui = Ui(self, self._aio_loop)

        self.downloads = DownloadManager()
        self.playlist = PlaylistManager() if play else None

        self.videos = dict()
        self.archive = Archive(archive_filename) if use_archive else None
        if self.archive:
            for video in self.archive:
                self._aio_loop.create_task(self.add_video(video))

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

    async def add_video(self, video):
        """Manage a video's progression through life."""
        if video.url in self.videos:
            original = self.videos[video.url]
            video.sync_to_original(original)
            self.ui.add_video(video)
            if self.playlist:
                await original.finished
                self.playlist.add_video(video)
            return

        self.videos[video.url] = video
        self.ui.add_video(video)
        if video.status == "pending":
            await self._aio_loop.run_in_executor(self._executor, video.prepare_meta)
            if os.path.isfile(video.filename):
                # multiple URLs pointing to the same video
                video.status == "duplicate"
        if video.status not in {"finished", "error", "duplicate"}:
            await self.downloads.download(video)
            if self.playlist:
                self.playlist.add_video(video)


if __name__ == "__main__":
    args = docopt(__doc__, version="0.0.1")
    ydl = YDL(play=args["--play"],
              resume=args["--continue"],
              use_archive=not args["--no-archive"],
              archive_filename=args["--archive-filename"])
    ydl.run()
