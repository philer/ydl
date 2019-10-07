#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import asyncio
import dataclasses
import logging
import os
import shutil
import signal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import partial
from typing import Iterable, Iterator, Optional
from urllib.parse import urlparse

import youtube_dl

from .ui import Ui

__title__ = "ydl"
__version__ = "0.1.0"
__license__ = "MIT"
__status__ = "Development"

__author__ = "Philipp Miller"
__email__ = "me@philer.org"
__copyright__ = "Copyright 2019 Philipp Miller"


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
