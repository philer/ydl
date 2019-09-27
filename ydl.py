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
import threading

from urwid import (AttrMap, Columns, Divider, Edit, ExitMainLoop, Frame,
                   ListBox, MainLoop, Pile, Text, WidgetWrap)
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

    def __init__(self, core):
        self._core = core
        self._lock = threading.Lock()
        self._videos = []
        self._worker_ids = iter(count())
        self._workers = dict()
        self._per_host = defaultdict(int)
        self._interrupted = False

    def shutdown(self):
        threads = set(self._workers.values())
        self._interrupted = True
        for thread in threads:
            thread.join()

    def add_video(self, video):
        with self._lock:
            self._videos.append(video)
            if len(self._workers) < MAX_POOL_SIZE:
                worker_id = next(self._worker_ids)
                thread = threading.Thread(target=self._work, args=(worker_id,))
                self._workers[worker_id] = thread
                thread.start()

    def _next(self, finished=None):
        with self._lock:
            if finished:
                self._per_host[finished.hostname] -= 1
            for i, video in enumerate(self._videos):
                if self._per_host[video.hostname] < MAX_HOST_POOL_SIZE:
                    self._per_host[video.hostname] += 1
                    del self._videos[i]
                    return video

    def _work(self, worker_id):
        """Perform actual downloads - this is run in a thread."""
        def progress_hook(data):
            """Called by youtube-dl periodically - used to break out when we want to quit."""
            if self._interrupted:
                raise Interrupt()
            video.set_info(**data)

        ydl = youtube_dl.YoutubeDL(dict(ydl_settings, progress_hooks=[progress_hook]))

        try:
            video = None
            while video := self._next(finished=video):
                video.set_info(status="downloading")
                try:
                    ydl.download([video.url])
                except youtube_dl.utils.DownloadError:
                    video.set_info(status="error")
        except Interrupt:
            pass
        del self._workers[worker_id]


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
            with self._ui.draw_lock:
                self._title_widget.set_text(f"{self.id} - {self.title}")

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

    def __init__(self, core):
        self._core = core

        self._input = Edit(caption=("prompt", "⟩ "))
        self._videos = ListBox([])
        footer = Pile([AttrMap(Divider("─"), "divider"), self._input])
        self._root = Frame(body=self._videos, footer=footer)
        self._root.focus_position = "footer"

        self._loop = MainLoop(widget=self._root,
                              palette=self.palette,
                              unhandled_input=self._handle_global_input)
        self._loop.screen.set_terminal_properties(
            colors=256, bright_is_bold=False, has_underline=True)

        self.draw_lock = threading.RLock()

    def run_loop(self):
        signal.signal(signal.SIGINT, self.halt_loop)  # should this be reset?
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
            if url:
                self._core.create_video(url)

class YDL:
    """Core controller"""
    def __init__(self):
        self.videos = dict()
        self.downloads = DownloadManager(self)
        self.ui = Ui(self)

        # stats = defaultdict(int)
        # print("Unfinished:")
        # for video in ...:
        #     stats[video.status] += 1
        #     if video.status in {"pending", "downloading"}:
        #         print(video.url)
        # print(f"Finished {stats['finished']}/{sum(stats.values())} jobs "
        #       f"({stats['pending']} pending, "
        #       f"{stats['downloading']} running, "
        #       f"{stats['error']} failed).")

    def run_loop(self):
        self.ui.run_loop()
        self.downloads.shutdown()

    def create_video(self, url):
        video = Video(self.ui, url)
        thread = threading.Thread(target=self._prepare_video, args=(video,))
        thread.start()
        self.ui.add_video(video)

    def _prepare_video(self, video):
        video.prepare_meta()  # blocks for a short while
        if video.id in self.videos:
            video.status = "duplicate"
        else:
            self.videos[video.id] = video
            self.downloads.add_video(video)

if __name__ == "__main__":
    YDL().run_loop()
