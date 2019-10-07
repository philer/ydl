#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""YDL - An interactive CLI for youtube-dl

Usage:
    ydl [-pc] [-a | --no-archive] [-v|-vv]
    ydl -h

Options:
    -p --play       Automatically play videos once they have
                    finished downloading
    -c --continue   Resume unfinishe download (existing .part/.ytdl files)
    -a --archive-filename
                    File to be used for storing status and URL of downloads
    --no-archive    Start a completely empty session, do not use an archive file
    -v --verbose   Show more and more info.
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
import asyncio
import dataclasses
import io
import logging
import os
import shutil
import signal
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import partial
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

from docopt import docopt
import urwid
from urwid import (AsyncioEventLoop, AttrMap, Columns, Divider, Edit,
                   ExitMainLoop, Filler, Frame, LineBox, ListBox, MainLoop,
                   Overlay, Padding, Pile, Text, WidgetPlaceholder, WidgetWrap)
import youtube_dl
import pyperclip

VIDEO_PLAYER = "/usr/bin/mpv"
VIDEO_DELAY = 0
ARCHIVE_FILENAME = ".ydl_archive"
MAX_POOL_SIZE = 5
MAX_HOST_POOL_SIZE = 3


log = logging.getLogger(__name__)

ydl_settings = {
    "format": "best",
    "noplaylist": True,
    "call_home": False,
    "cachedir": False,
    # "download_archive": ".youtube_dl_archive",
    "logger": logging.getLogger("youtube_dl"),
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
        self._original = original
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

    async def play(self):
        try:
            with open(os.devnull, 'w') as devnull:
                process = await asyncio.create_subprocess_exec(VIDEO_PLAYER,
                                                               self.filename,
                                                               stdout=devnull,
                                                               stderr=devnull)
                await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            raise
        except Exception as e:
            log.exception(e)

    def delete(self):
        """
        Remove this video file from the file system and mark it as deleted.
        TODO: Also interrupt any pending or ongoing download and cleanup temporary files.
        """
        if self.status in {"deleted", "error"}:
            return
        elif self.status == "duplicate":
            self._original.delete()
        elif self.status == "pending":
            ...  # TODO interrupt
        elif self.status == "downloading":
            ...  # TODO interrupt
        else:
            log.info(f"Removing file '{self.filename}'.")
            os.remove(self.filename)  # TODO some error handling on this
            self.status = "deleted"


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
                log.info(f"Creating backup of .ydl_archive at {backup}")
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
        asyncio.ensure_future(self._run())

    def add_video(self, video):
        asyncio.create_task(self._playlist.put(video))

    async def _run(self):
        while True:
            await (await self._playlist.get()).play()
            self._playlist.task_done()
            await asyncio.sleep(self.delay)


class VideoWidget(WidgetWrap):
    """Ugly mix of data model and view widget"""

    palette = [
        ("pending",            "light gray",  "", "", "g70",  ""),
        ("duplicate",          "yellow",      "", "", "#da0", ""),
        ("downloading",        "light blue",  "", "", "#6dd", ""),
        ("downloading_filled", "standout",    "", "", "g0",   "#6dd"),
        ("finished",           "",            "", "", "",     ""),
        ("finished_icon",      "light green", "", "", "#8f6", ""),
        ("error",              "light red",   "", "", "#d66", ""),
        ("deleted",            "dark gray,strikethrough", "", "", "#666,strikethrough", ""),
        ("deleted_icon",       "dark gray",   "", "", "#666", ""),
    ]
    palette += [(p[0] + "_focus", p[1] + ",bold", *p[2:4], p[4] + ",bold", p[5])
                for p in palette if not p[0].endswith("_icon")]

    status_icon = {
        "pending": " ⧗ ",
        "duplicate": " = ",
        "downloading": " ⬇ ",
        "finished": " ✔ ",
        "error": " ⨯ ",
        "deleted": " ⨯ ",
        # ▶ ♻
    }

    @property
    def _info(self):
        if self._video.id is not None and self._video.title is not None:
            return f"{self._video.id} - {self._video.title}"
        return self._video.url

    def __init__(self, ui, video):
        self._ui = ui
        self._video = video

        self._status_widget = Text(" ? ")
        self.update_status_icon()
        self._info_widget = Text("", wrap='clip')
        self._divider = Text(("divider", "│"))
        columns = [
            (3, self._status_widget),
            (1, self._divider),
            self._info_widget,
        ]
        self._root = Columns(columns, dividechars=1)
        super().__init__(self._root)

        video.observers.append(self._handle_update)

    def _handle_update(self, _video, prop, _value):
        if prop in {"status", "progress"}:
            self.update_status_icon()
        with self._ui.draw_lock:
            self._ui._loop.draw_screen()

    def update_status_icon(self):
        status = self._video.status
        style = status + "_icon" if status in {"finished", "deleted"} else status
        if status == "downloading" and self._video.progress:
            icon = f"{self._video.progress: >3.0%}"
        else:
            icon = self.status_icon[status]
        with self._ui.draw_lock:
            self._status_widget.set_text((style, icon))

    async def _delete(self):
        if await self._ui.confirm("Really delete?"):
            self._video.delete()

    def selectable(self):
        return True

    def keypress(self, _size, key):
        if key == "p" or key == "enter":
            self._ui._aio_loop.create_task(self._video.play())
        elif key == "d" or key == "delete":
            self._ui._aio_loop.create_task(self._delete())
        else:
            return key

    def render(self, size, focus=False):
        """Update appearance based on video status, widget focus and size."""
        status = self._video.status
        focused = "_focus" if focus else ""
        info = self._info.ljust(size[0])
        if status == "downloading":
            filled_width = int(size[0] * self._video.progress)
            filled, empty = info[:filled_width], info[filled_width:]
            self._info_widget.set_text([("downloading_filled" + focused, filled),
                                        ("downloading" + focused, empty)])
        else:
            self._info_widget.set_text((status + focused, info))
        self._divider.set_text(("divider_focus", "┃") if focus else ("divider", "│"))  # ╽╿
        self._invalidate()
        return self._root.render(size, focus)


class Button(WidgetWrap):
    signals = ["click"]

    def __init__(self, caption, callback=None):
        root = AttrMap(Text(f"[{caption}]"), "button", "button_focus")
        super().__init__(root)
        if callback:
            urwid.connect_signal(self, "click", callback)

    def selectable(self):
        return True

    def keypress(self, _size, key):
        """Send "click" signal on 'activate' command."""
        if self._command_map[key] == urwid.ACTIVATE:
            self._emit("click")
        else:
            return key

    def mouse_event(self, _size, event, button, *_):
        """Send "click" signal on button 1 press."""
        if button == 1 and urwid.util.is_mouse_press(event):
            self._emit("click")
            return True
        return False

class Dialog(WidgetWrap):
    """
    Dialog Wídget that can be attached to an existing WidgetPlaceholder.
    As a (experimental) subclass of asyncio.Future the result can be awaited.
    """
    def __init__(self, content, *, parent=None, buttons=("cancel", "continue")):
        self._parent = parent
        self._future = asyncio.get_event_loop().create_future()
        if isinstance(content, str):
            content = Text(content, align="center")
        elif not isinstance(content, urwid.Widget):
            raise TypeError("Content of Dialog widget must be instance of Widget or str.")
        if buttons:
            row = [Divider()]
            for button in buttons:
                if isinstance(button, str):
                    button, name = Button(button.capitalize()), button
                    urwid.connect_signal(button, "click", self.close, user_args=[name])
                row.append(('pack', button))
            row.append(Divider())
            row = Columns(row, dividechars=3)
            row.focus_position = len(buttons)
            row = Padding(row, align="center", width="pack")
            content = Pile([content, Divider(top=1), row])
        self._root = LineBox(Filler(content))
        super().__init__(self._root)
        if parent:
            self.show(parent)

    def __getattr__(self, attr):
        try:
            return getattr(self._future, attr)
        except AttributeError:
            raise AttributeError from None

    def __await__(self):
        return self._future.__await__()

    def show(self, parent):
        assert isinstance(parent, WidgetPlaceholder)
        self._parent = parent
        widget = Overlay(self,
                         parent.original_widget,
                         align='center', valign='middle',
                         height=('relative', 50), min_height=5,
                         width=('relative', 50), min_width=10)
        parent.original_widget = widget
        parent._invalidate()

    def close(self, result=None, *_):
        if self._parent:
            self._parent.original_widget = self._parent.original_widget.contents[0][0]
        if result == "cancel":
            self._future.cancel()
        else:
            self._future.set_result(result)

    def keypress(self, size, key):
        """Swallow anything the Dialog content doesn't use."""
        if self._root.keypress(size, key) is None:
            return
        if key == "esc":
            self.close("cancel")
        else:
            return key

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
                self._exc_info = sys.exc_info()
                if self._exc_info == (None, None, None):
                    self._exc_info = (type(exc), exc, exc.__traceback__)
        else:
            loop.default_exception_handler(context)

class Ui:
    palette = (
        *VideoWidget.palette,
        ("divider",         "dark gray",   "", "", "#666", ""),
        ("divider_focus",   "light gray",  "", "", "#aaa", ""),
        ("prompt",          "light green", ""),
        ("button",          "bold",        ""),
        ("button_focus",    "bold,standout",""),
    )

    def __init__(self, core, aio_loop):
        self._core = core
        self._aio_loop = aio_loop

        self._input = Edit(caption=("prompt", "⟩ "))
        self._videos = ListBox([])
        footer = Pile([AttrMap(Divider("─"), "divider"), self._input])
        self._main = Frame(body=self._videos, footer=footer)
        self._root = WidgetPlaceholder(self._main)

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
        if isinstance(key, tuple) and key[0] == "mouse press":
            _event, button, _x, _y = key
            if 4 <= button <= 5:
                if self._main.focus_position == "body":
                    try:
                        if button == 4:
                            self._videos.focus_position -= 1
                        else:
                            self._videos.focus_position += 1
                    except IndexError:
                        pass
                else:
                    self._main.focus_position = "body"
                    if button == 4:
                        try:
                            self._videos.focus_position = len(self._videos.body) - 1
                        except IndexError:
                            pass

        elif key == "q" or key == "esc":
            self._core.shutdown()
        elif key == "ctrl v":
            try:
                self._handle_urls(pyperclip.paste())
            except pyperclip.PyperclipException:
                pass
        elif key == "enter" and self._input.edit_text:
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
            self._videos.focus_position = len(self._videos.body) - 1

    async def confirm(self, message):
        return "continue" == await Dialog(message, parent=self._root)


class YDL:
    """Core controller"""
    def __init__(self, archive=None, play=False, resume=False):
        self.archive = archive
        self._aio_loop = asyncio.get_event_loop()
        self._executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self.ui = Ui(self, self._aio_loop)

        self.downloads = DownloadManager()
        self.playlist = PlaylistManager() if play else None

        self.videos = dict()
        if archive:
            for video in archive:
                self._aio_loop.create_task(self.add_video(video))

    def run(self):
        self._aio_loop.add_signal_handler(signal.SIGINT, self.shutdown)
        self.ui.run_loop()
        self.downloads.shutdown()
        self._aio_loop.run_until_complete(self._cleanup_tasks())
        self._aio_loop.close()
        self.archive.update(self.videos.values())

    def shutdown(self):
        self.ui.halt_loop()

    async def _cleanup_tasks(self):
        tasks = asyncio.all_tasks(self._aio_loop)
        current_task = asyncio.current_task(self._aio_loop)
        tasks = {task for task in tasks if task is not current_task}
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

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
            if video.status != "error" and os.path.isfile(video.filename):
                # multiple URLs pointing to the same video
                video.status == "duplicate"
        if video.status not in {"finished", "error", "deleted", "duplicate"}:
            await self.downloads.download(video)
            if self.playlist:
                self.playlist.add_video(video)


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

if __name__ == "__main__":
    args = docopt(__doc__, version="0.0.1")
    log_stream = setup_logging(args["--verbose"])
    archive = None if args["--no-archive"] else Archive(args["--archive-filename"])
    ydl = YDL(archive, play=args["--play"], resume=args["--continue"])
    try:
        ydl.run()
    finally:
        print(log_stream.getvalue(), end="")
