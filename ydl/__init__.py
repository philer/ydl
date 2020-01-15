#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import asyncio
import logging
import os
import random
import shutil
import signal
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import partial, wraps
from typing import Any, Iterable, Iterator, Optional
from urllib.parse import urlparse

import youtube_dl
import requests

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


@dataclass
class Video:
    """Primary model."""
    url: str
    status: str = "pending"
    progress: int = 0
    finished: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    _meta_properties = "extractor", "id", "title", "ext"
    extractor: str = None
    id: str = None
    title: str = "(no title)"
    ext: str = None

    _original: Video = None
    observers: List[callable] = field(default_factory=list)

    def original(method):
        """Method decorator to only call state mutating methods
        of duplicates on original."""
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self._original:
                return method(self._original, *args, **kwargs)
            return method(self, *args, **kwargs)
        return wrapped

    @original
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

    def sync_to_original(self, original):
        """
        This video has been identified as a duplicate
        and should have its meta info mirror that of the original.
        """
        self.status = "duplicate"
        for prop in self._meta_properties:
            setattr(self, prop, getattr(original, prop))
        original.observers.append(self._handle_update)
        self._original = original

    def _handle_update(self, original, prop, value):
        """Update a meta property mirrored from another instance."""
        if prop in self._meta_properties:
            setattr(self, prop, value)

    @original
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

    @original
    async def play(self):
        """Play the video in an external video player."""
        if self.status == "finished":
            filename = self.filename
        elif self.status == "downloading":
            filename = self.filename + ".part"
        else:
            return

        try:
            process = await asyncio.create_subprocess_exec(VIDEO_PLAYER,
                                                           "--fullscreen",
                                                           filename,
                                                           stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.DEVNULL)
            await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            raise
        except Exception as e:
            log.exception(e)

    @original
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
            try:
                os.remove(self.filename)
            except OSError as ose:
                log.warning(ose)
            self.status = "deleted"

    del original


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
                        if len(parts) in {2, 5, 6}:
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
                if video.status == "pending":
                    archive.write(f"{video.status} {video.url}\n")
                elif video.status not in {"error", "duplicate"}:
                    parts = (getattr(video, prop) for prop in self._video_properties)
                    archive.write(" ".join(parts) + "\n")


class ThreadsafeProxy:
    """
    Wrap an object so all method calls and attribute assignments are deferred
    to asyncio's call_soon_threadsafe.

    Limitations:
    - Property get-access is _not_ deferred. That means if an @property's
      getter does significant non-threadsafe work it will need extra treatment.
    - Any callable property is treated as a method and wrapped on access.
    """

    # This is a very basic proxy implementation. If additional capabilites
    # become necessary in the future, this may be useful:
    # https://code.activestate.com/recipes/496741-object-proxying/

    def __init__(self, instance, loop=None):
        call = (loop or asyncio.get_event_loop()).call_soon_threadsafe
        object.__setattr__(self, "_instance", instance)
        object.__setattr__(self, "_call", call)

    def __getattr__(self, name):
        if callable(value := getattr(self._instance, name)):
            return partial(self._call, value)
        return value

    def __setattr__(self, name, value):
        self._call(setattr, self._instance, name, value)

    def __delattr__(self, name):
        self._call(delattr, self._instance, name)

    def __dir__(self):
        return dir(self._instance)

class Interrupt(Exception):
    """Used to break out of youtube-dl's blocking download via hook."""
    pass

class DownloadManager:
    """Manages video download workers and parallel connections per host."""

    def __init__(self, aio_loop):
        self._aio_loop = aio_loop
        self._meta_executor = ThreadPoolExecutor(max_workers=MAX_POOL_SIZE)
        self._download_executors = defaultdict(partial(ThreadPoolExecutor,
                                                       max_workers=MAX_HOST_POOL_SIZE))
        self._max_workers_sem = asyncio.BoundedSemaphore(value=MAX_POOL_SIZE)
        self._interrupted = False

    def shutdown(self):
        """Interrupt all running threads and wait for them to return."""
        self._interrupted = True
        for ex in self._download_executors.values():
            ex.shutdown(wait=True)

    def _raise_interrupt(self, _: Any):
        """
        Terminate a worker thread when we want to quit.
        This will be called periodically via youtube-dl's download feedback hook.
        """
        if self._interrupted:
            raise Interrupt

    async def download_meta(self, video: Video):
        """
        Download meta data for a given video. Some of a Video instance's
        functionality will not be available until this is done.
        """
        proxy = ThreadsafeProxy(video, loop=self._aio_loop)
        await self._aio_loop.run_in_executor(self._meta_executor,
                                             self._download_meta, proxy)

    def _download_meta(self, video: ThreadsafeProxy):
        """Perform actual meta info download - this should run in a thread."""
        try:
            # resolve/simplify URL
            response = requests.head(video.url)
            video.url = response.url
        except requests.exceptions.RequestException:
            video.status = "error"
            return

        ydl = youtube_dl.YoutubeDL(ydl_settings)
        try:
            meta = ydl.extract_info(video.url, download=False)
        except youtube_dl.utils.DownloadError:
            video.status = "error"
        else:
            for prop in Video._meta_properties:
                setattr(video, prop, meta.get(prop))
            filename_props = {"extractor", "id", "title", "ext"}
            if set(meta.keys()) & filename_props != filename_props:
                video.status = "error"

    async def download(self, video: Video):
        """Download a given video as soon as a slot is open."""
        async with self._max_workers_sem:
            if self._interrupted:
                return
            ex = self._download_executors[video.hostname]
            proxy = ThreadsafeProxy(video, loop=self._aio_loop)
            await self._aio_loop.run_in_executor(ex, self._download, proxy)

    def _download(self, video: ThreadsafeProxy):
        """Perform actual downloads - this should run in a thread."""
        hooks = [self._raise_interrupt, video.set_download_info]
        ydl = youtube_dl.YoutubeDL(dict(ydl_settings, progress_hooks=hooks))
        video.status = "downloading"
        try:
            ydl.download([video.url])
        except youtube_dl.utils.DownloadError:
            video.status = "error"
        except self.Interrupt:
            pass


class YDL:
    """Core controller"""
    def __init__(self, archive=None, play=False):
        self.archive = archive
        self._aio_loop = asyncio.get_event_loop()
        self.ui = Ui(self, self._aio_loop)

        self.downloads = DownloadManager(aio_loop = self._aio_loop)

        self.videos = dict()
        if archive:
            for video in archive:
                self.videos[video.url] = video
                self.ui.add_video(video)
                if video.status in {"pending", "downloading"}:
                    self._aio_loop.create_task(self._handle_new_video(video))

        self.playlist = asyncio.Queue()
        self._playlist_task = self._random_playlist_task = None
        if play:
            self.start_playlist()

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

    async def add_video(self, url: str):
        """Manage a video's progression through life."""
        video = Video(url)
        self.ui.add_video(video)
        if url in self.videos:
            original = self.videos[url]
            video.sync_to_original(original)
            await original.finished
            await self.playlist.put(video)
        else:
            self.videos[url] = video
            await self._handle_new_video(video)

    async def _handle_new_video(self, video):
        if video.status == "pending":
            await self.downloads.download_meta(video)
            if video.status != "error" and os.path.isfile(video.filename):
                # multiple URLs pointing to the same video
                video.status == "duplicate"
        if video.status in {"pending", "downloading"}:
            await self.downloads.download(video)
            await self.playlist.put(video)

    def start_playlist(self):
        if self._playlist_task is None:
            log.debug("Starting queue playback")
            self._playlist_task = self._aio_loop.create_task(self._run_playlist())

    def stop_playlist(self):
        if self._playlist_task:
            log.debug("Stopping queue playback")
            self._playlist_task.cancel()
            self._playlist_task = None

    async def _run_playlist(self):
        while True:
            await (await self.playlist.get()).play()
            self.playlist.task_done()
            await asyncio.sleep(VIDEO_DELAY)

    def start_random_playlist(self):
        if self._random_playlist_task is None:
            log.debug("Starting random playback")
            self._random_playlist_task = self._aio_loop.create_task(self._run_random_playlist())

    def stop_random_playlist(self):
        if self._random_playlist_task:
            log.debug("Starting random playback")
            self._random_playlist_task.cancel()
            self._random_playlist_task = None

    async def _run_random_playlist(self):
        """Continuously play random videos."""
        while True:
            await random.choice(tuple(self.videos.values())).play()
            await asyncio.sleep(VIDEO_DELAY)
