#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
test vine (spider):
https://www.youtube.com/watch?v=RwJe8KfPCEQ


TODO:
- tests
- delete option
- video playback:
    + finish order vs. adding order
    + delete option after play
    + play arbitrary (separate of queue)
    + countdown timer for next playback
"""

import os
import sys
import signal
from collections import defaultdict
from itertools import count
from urllib.parse import urlparse

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from urwid import (AttrMap, Columns, Divider, Edit, ExitMainLoop, Frame,
                   ListBox, MainLoop, Pile, Text, WidgetWrap,
                   AsyncioEventLoop)
import youtube_dl
import pyperclip


MAX_POOL_SIZE = 5
MAX_HOST_POOL_SIZE = 3


class MuteLogger:
    def noop(self, msg):
        pass
    debug = warning = error = noop

ydl_settings = {
    "format": "best",
    "noplaylist": True,
    "call_home": False,
    "cachedir": False,
    # "download_archive": "youtube_dl_archive",
    "logger": MuteLogger(),
    "outtmpl": "%(extractor)s-%(id)s_%(title)s.%(ext)s"
    # "outtmpl": "/tmp/ydl/" + youtube_dl.DEFAULT_OUTTMPL,
}

class Interrupt(Exception):
    """Used to break out of youtube-dl's blocking download via hook."""
    pass

class DownloadManager:
    """Manages video download workers and parallel connections per host."""

    def __init__(self):
        self._videos = []
        self._executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self._lock = threading.Lock()
        self._per_host = defaultdict(int)
        self._interrupted = False

    def shutdown(self):
        self._interrupted = True
        self._executor.shutdown(wait=True)

    async def download(self, video):
        done = asyncio.Event()
        with self._lock:
            self._videos.append((video, done))

        # submit one job. it may not be the one taking care of this video.
        future = self._executor.submit(self._work)
        if e := future.exception():
            raise e
        await done.wait()

    def _next(self):
        with self._lock:
            for i, (video, done) in enumerate(self._videos):
                if self._per_host[video.hostname] < MAX_HOST_POOL_SIZE:
                    self._per_host[video.hostname] += 1
                    del self._videos[i]
                    return video, done

    def _finished(self, video):
        with self._lock:
            self._per_host[video.hostname] -= 1

    def _work(self):
        """Perform actual downloads - this is run in a thread."""

        # TODO move setup code to ThreadPoolExecutor initializer
        def progress_hook(data):
            """Called by youtube-dl periodically - used to break out when we want to quit."""
            if self._interrupted:
                raise Interrupt()
            video.set_info(**data)

        ydl = youtube_dl.YoutubeDL(dict(ydl_settings, progress_hooks=[progress_hook]))

        video, done = self._next()
        video.set_info(status="downloading")
        try:
            ydl.download([video.url])
        except youtube_dl.utils.DownloadError:
            video.set_info(status="error")
        except Interrupt:
            return
        self._finished(video)
        done.set()  # No idea if this is okay. asyncio.Event is not threadsafe.


class PlaylistManager:
    def __init__(self, delay=3):
        self.delay = delay
        self._videos = asyncio.Queue()
        self._interrupted = False

    async def add_video(self, video):
        await self._videos.put(video)

    def start(self):
        asyncio.get_event_loop().create_task(self._run())

    def stop(self):
        self._interrupted = True

    async def _run(self):
        while not self._interrupted:
            video = await self._videos.get()
            devnull = open(os.devnull, 'w')
            process = await asyncio.create_subprocess_exec("mpv", video.filename,
                                                           stdout=devnull,
                                                           stderr=devnull)
            await process.wait()
            self._videos.task_done()
            await asyncio.sleep(self.delay)


class Video(WidgetWrap):
    """Ugly mix of data model and view widget"""
    status_icon = {
        "pending": " ⧗ ",
        "duplicate": " = ",
        "downloading": " ⬇ ",
        "finished": " ✔ ",
        "error": " ⨯ ",
    }

    @property
    def hostname(self):
        return urlparse(self.url).hostname

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

    def __init__(self, ui, url):
        self._ui = ui
        self.url = url
        self.progress = 0
        self._status = "pending"

        self._status_widget = Text(self.status_icon["pending"])
        self._title_widget = Text(url, wrap='clip')
        columns = [
            (3, self._status_widget),
            (1, Text(("divider", "│"))),
            self._title_widget,
        ]
        self._root = AttrMap(Columns(columns, dividechars=1), "pending")
        super().__init__(self._root)

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
            self.id = self.url
        else:
            for key in "id", "title", "ext":
                self.__dict__.setdefault(key, meta[key])
            self.filename = "{extractor}-{id}_{title}.{ext}".format_map(meta)
            with self._ui.draw_lock:
                self._title_widget.set_text(f"{self.id} - {self.title}")
                self._ui._loop.draw_screen()

    def set_info(self, status, downloaded_bytes=0, total_bytes=None,
                 total_bytes_estimate=None, **kwargs):
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
        if total_bytes is None and total_bytes_estimate is not None:
            total_bytes = total_bytes_estimate
        if status == "downloading" and total_bytes:
            self.progress = downloaded_bytes / total_bytes
        else:
            self.progress = 0
        self.status = status

    def render(self, size, focus=False):
        """hack allows me to change text background based on size"""
        title_text = self._title_widget.text.strip().ljust(size[0])
        filled_width = int(size[0] * self.progress)
        filled, empty = title_text[:filled_width], title_text[filled_width:]
        self._title_widget.set_text([("progress_filled", filled), empty])
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
            self.halt_loop()
        elif key == 'ctrl v':
            self._handle_urls(pyperclip.paste())
        elif key == 'enter' and self._input.edit_text:
            self._handle_urls(self._input.edit_text)
            self._input.edit_text = ""

    def _handle_urls(self, text):
        """Extract valid urls from user input and pass them on."""
        for url in map(str.strip, text.split(sep="\n")):
            if url:  # maybe check URL with a regex?
                self._aio_loop.create_task(self._core.create_video(url))

class YDL:
    """Core controller"""
    def __init__(self):
        self.videos = dict()
        self._aio_loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self.ui = Ui(self, self._aio_loop)
        self.downloads = DownloadManager()
        self.playlist = PlaylistManager()

    def run_loop(self):
        self._aio_loop.add_signal_handler(signal.SIGINT, self.ui.halt_loop)
        try:
            self.playlist.start()
            self.ui.run_loop()
        finally:
            self._aio_loop.remove_signal_handler(signal.SIGINT)
            self.playlist.stop()
            self.downloads.shutdown()

    async def create_video(self, url):
        video = Video(self.ui, url)
        self.ui.add_video(video)
        await self._aio_loop.run_in_executor(self._executor, video.prepare_meta)
        if video.status == "error":
            return
        if video.id in self.videos:
            video.status = "duplicate"
        else:
            self.videos[video.id] = video
            await self.downloads.download(video)
        await self.playlist.add_video(video)


if __name__ == "__main__":
    YDL().run_loop()
