#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
test vine (spider):
https://www.youtube.com/watch?v=RwJe8KfPCEQ


TODO:
download:
    - 1 pool per host (done)
    - maximum parallel downloads overall
playback:
    - finish order vs. adding order
    - delete option after play
    - play/delete arbitrary
    - countdown timer for next playback
"""

import os
import sys
import signal
from collections import defaultdict

from urllib.parse import urlparse
from pathlib import Path

import threading
from queue import SimpleQueue

import urwid
from urwid import (AttrMap, Columns, Divider, Edit, ExitMainLoop, Frame,
                   ListBox, MainLoop, Pile, Text, WidgetWrap)
import youtube_dl


POOL_SIZE = 3


class MuteLogger:
    def noop(self, msg):
        pass
    debug = warning = error = noop

ydl_settings = {
    "format": "best",
    "noplaylist": True,
    "logger": MuteLogger(),
    "outtmpl": "%(extractor)s-%(id)s_%(title)s.%(ext)s"
    # "outtmpl": Path("~/ydl").expanduser() / youtube_dl.DEFAULT_OUTTMPL,
    # "outtmpl": "/tmp/ydl/" + youtube_dl.DEFAULT_OUTTMPL,
}


class Pool:
    def __init__(self, size=POOL_SIZE):
        self.queue = SimpleQueue()
        self.workers = []
        for _ in range(size):
            worker = Worker(self.queue)
            worker.start()
            self.workers.append(worker)

    def add_job(self, job):
        self.queue.put(job)

    def shutdown(self):
        for worker in self.workers:
            worker.interrupt()
            self.queue.put(None)  # resolve blocking queue.get()

    def join(self):
        for worker in self.workers:
            worker.join()


class Worker(threading.Thread):
    def __init__(self, queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self._ydl = youtube_dl.YoutubeDL(dict(ydl_settings, progress_hooks=[self._progress_hook]))
        self._current_job = None
        self._interrupted = False
        self.working = False

    def interrupt(self):
        self._interrupted = True

    class Interrupt(Exception):
        """Used to break out of youtube-dl's blocking download via hook."""
        pass

    def _progress_hook(self, data):
        if self._interrupted:
            raise self.Interrupt()
        else:
            self._current_job.set_info(**data)

    def run(self):
        self._current_job = self.queue.get()
        while not self._interrupted:
            self.working = True
            self._current_job.set_info(status="downloading")
            try:
                self._ydl.download([self._current_job.url])
            except self.Interrupt:
                break
            except youtube_dl.utils.DownloadError:
                self._current_job.set_info(status="error")
            self.working = False
            self._current_job = self.queue.get()


class Job(WidgetWrap):
    """Nasty mix of job data store and ui widget."""

    _instances = dict()

    status_icon = {
        "pending": " ⧗ ",
        "duplicate": " 2 ",
        "downloading": " ⬇ ",
        "finished": " ✔ ",
        "error": " ⨯ ",
    }

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

    def __init__(self, pool, ui, url):
        self._pool = pool
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
        threading.Thread(target=self._prepare).start()

    def _prepare(self):
        ydl = youtube_dl.YoutubeDL(ydl_settings)
        try:
            meta = ydl.extract_info(self.url, download=False)
        except youtube_dl.utils.DownloadError:
            self.status = "error"
            self._instances[self.url] = self
            return

        duplicate = meta["id"] in self._instances
        if duplicate:
            self.status = "duplicate"
        else:
            self._instances[meta["id"]] = self
            self._pool.add_job(self)
        with self._ui.draw_lock:
            self._title_widget.set_text(f"{meta['id']} - {meta['title']}")

    def set_info(self, status, downloaded_bytes=0, total_bytes=None,
                 total_bytes_estimate=None, **kwargs):
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

    def __init__(self):
        self._input = Edit(caption=("prompt", "⟩ "))
        self._downloads = ListBox([])
        footer = Pile([AttrMap(Divider("─"), "divider"), self._input])
        self._root = Frame(body=self._downloads, footer=footer)
        self._root.focus_position = "footer"
        self._loop = MainLoop(widget=self._root,
                              palette=self.palette,
                              unhandled_input=self._handle_global_input)
        self._loop.screen.set_terminal_properties(
            colors=256, bright_is_bold=False, has_underline=True)

        self.draw_lock = threading.RLock()

    def run_loop(self):
        signal.signal(signal.SIGINT, self.halt_loop)  # should this be reset?
        self._pools = defaultdict(Pool)
        self._loop.run()

    def halt_loop(self, *_):
        for pool in self._pools.values():
            pool.shutdown()
        for pool in self._pools.values():
            pool.join()
        raise ExitMainLoop()

    def _handle_global_input(self, key):
        if key == 'esc':
            self.halt_loop()
        elif key == 'enter' and self._input.edit_text:
            for url in map(str.strip, self._input.edit_text.split(sep="\n")):
                if url:
                    hostname = urlparse(url).hostname
                    job = Job(self._pools[hostname], self, url)
                    with self.draw_lock:
                        self._downloads.body.append(job)
            self._input.edit_text = ""


if __name__ == "__main__":

    ui = Ui()
    ui.run_loop()

    stats = {s: 0 for s in ("pending", "downloading", "finished",  "error")}
    print("Unfinished:")
    for job in Job._instances.values():
        stats[job.status] += 1
        if job.status in {"pending", "downloading"}:
            print(job.url)
    print(f"Finished {stats['finished']}/{sum(stats.values())} jobs "
          f"({stats['pending']} pending, "
          f"{stats['downloading']} running, "
          f"{stats['error']} failed).")
