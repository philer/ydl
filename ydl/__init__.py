#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import asyncio
import logging
import os
import random
import re
import shutil
import signal
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from functools import partial, wraps
from pathlib import Path
from typing import (cast, Any, Callable, Dict, Iterable, Iterator, List,
                    Optional, Sequence, TypeVar)
from urllib.parse import urlparse

import youtube_dl  # type: ignore
import requests

from .util import noawait, Observable, ThreadsafeProxy
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

_youtube_dl_logger = logging.getLogger(__package__ + ".youtube_dl")
_youtube_dl_logger.setLevel(logging.INFO)
ydl_settings = {
    "format": "best",
    "noplaylist": True,
    "call_home": False,
    "cachedir": False,
    # "download_archive": ".youtube_dl_archive",
    "logger": _youtube_dl_logger,
    "outtmpl": "%(extractor)s-%(id)s_%(title)s.%(ext)s"
    # "outtmpl": "/tmp/ydl/" + youtube_dl.DEFAULT_OUTTMPL,
}

F = TypeVar('F', bound=Callable[..., Any])

@dataclass(eq=False)
class Video(Observable):
    """Primary model."""
    url: str
    status: str = "pending"
    progress: int = 0
    finished: asyncio.Future = field(init=False, repr=False, compare=False,
                                     default_factory=lambda: asyncio.get_event_loop().create_future())

    _meta_properties = "extractor", "id", "title", "ext"
    extractor: str = ""
    id: str = ""
    title: str = ""
    ext: str = ""

    _original: Optional[Video] = field(init=False, repr=False, compare=False,
                                       default=None)
    playing: int = field(init=False, repr=False, compare=False, default=0)

    def __post_init__(self):
        if not self.title:
            self.title = self.url

    def original(method: F) -> F:  # type: ignore
        """Method decorator to only call state mutating methods
        of duplicates on original."""
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if self._original:
                return method(self._original, *args, **kwargs)
            return method(self, *args, **kwargs)
        return cast(F, wrapped)

    @original
    def __setattr__(self, name, value):
        """A simplistic observer pattern."""
        super().__setattr__(name, value)

    @property
    def hostname(self):
        return urlparse(self.url).hostname

    _re_unsafe_characters = re.compile(r"(?:_|[^\w()[\]])+")

    @property
    def filename(self):
        if not all((self.extractor, self.id, self.ext)):
            raise AttributeError(f"Can't generate filename for {self}.")
        title = self._re_unsafe_characters.sub("_", self.title).strip("_")
        return f"{self.extractor}-{self.id}_{title}.{self.ext}"

    def _find_real_file(self, rename=True):
        """Find a matching file in the file system and rename it appropriately."""
        pattern = f"{self.extractor}-{self.id}*"
        expected = Path(self.filename)
        paths = set(Path(".").glob(pattern))
        if not paths:
            raise RuntimeError(f"Found no files matching '{pattern}'.")
        try:
            path, = paths
        except ValueError:
            if expected in paths:
                log.warning("Found multiple paths match '%s'.", pattern)
                return expected
            raise RuntimeError(f"Too many files to choose from for '{pattern}'.")
        if path == expected:
            return expected
        if rename:
            log.info(f"Renaming file '{path}' -> '{expected}'.")
            path.rename(expected)
            return expected
        return path

    def sync_to_original(self, original: Video):
        """
        This video has been identified as a duplicate
        and should have its meta info mirror that of the original.
        """
        self.status = "duplicate"
        for prop in self._meta_properties:
            setattr(self, prop, getattr(original, prop))
        original.subscribe(self._handle_update)
        self._original = original

    def _handle_update(self, original: Video, prop: str, value: Any):
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
            filename = self._find_real_file()
        elif self.status == "downloading":
            filename = self.filename + ".part"
        else:
            return
        log.info("Playing %s", filename)
        self.playing += 1
        try:
            process = await asyncio.create_subprocess_exec(VIDEO_PLAYER,
                                                           "--fullscreen",
                                                           filename,
                                                           stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.DEVNULL)
            if 0 != await process.wait():
                raise Exception(f"Playing '{self.filename}' failed.")
        except asyncio.CancelledError:
            process.terminate()
            raise
        except Exception as e:
            log.exception(e)
        finally:
            self.playing -= 1

    @original
    def delete(self):
        """
        Remove this video file from the file system and mark it as deleted.
        TODO: Also interrupt any pending or ongoing download and cleanup temporary files.
        """
        if self.status in {"deleted", "error"}:
            log.info("Nothing to delete.")
        elif self.status == "duplicate":
            self._original.delete()
        elif self.status in {"pending", "downloading"}:
            # TOOD implement interrupt & delete
            raise NotImplementedError("Can't delete pending/ongoing downloads (yet).")
        else:
            pattern = f"{self.extractor}-{self.id}*"
            paths = tuple(Path(".").glob(pattern))
            for path in paths:
                log.info(f"Removing file '{path}'")
                try:
                    path.unlink()
                except OSError as ose:
                    log.exception(ose)
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
                        parts: List[Any] = line.split(None, len(self._video_properties) - 1)
                        if len(parts) not in {2, 5, 6}:
                            raise ValueError(f"Invalid archive state on line {lineno}: '{line}'")
                        yield Video(**dict(zip(self._video_properties, parts)))
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

        log.info("Writing archive...")
        with open(self._filename, "wt") as archive:
            for video in videos:
                if video.status == "pending":
                    archive.write(f"{video.status} {video.url}\n")
                elif video.status not in {"error", "duplicate"}:
                    parts = (getattr(video, prop) for prop in self._video_properties)
                    archive.write(" ".join(parts) + "\n")


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
            ex.shutdown(wait=False)

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
        ydl = youtube_dl.YoutubeDL(dict(ydl_settings,
                                        outtmpl=video.filename,
                                        progress_hooks=hooks))
        video.status = "downloading"
        try:
            ydl.download([video.url])
        except youtube_dl.utils.DownloadError:
            video.status = "error"
        except Interrupt:
            pass


class Playlist(Observable):

    delay = 0

    def __init__(self, videos):
        super().__init__()
        self.videos = videos
        self.current = None
        self._iter = iter(videos)
        self._task = None

    def start(self):
        if self._task is None:
            log.debug("Starting playlist")
            self._task = noawait(self._run())

    def stop(self):
        if self._task:
            log.debug("Stopping playlist")
            self._task.cancel()
            self._task = None
            self.current = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        return next(self._iter)

    async def _run(self):
        try:
            async for self.current in self:
                await self.current.play()
                await asyncio.sleep(self.delay)
        except Exception as e:
            log.exception(e)

class RandomPlaylist(Playlist):
    def __init__(self, videos: Sequence[Video]):
        super().__init__([])
        self._source = videos

    async def __anext__(self):
        video = random.choice(self._source)
        self.videos = self.videos + [video]
        return video

class QueuePlaylist(Playlist):
    def __init__(self, videos: asyncio.Queue[Video]):
        super().__init__([])
        self._queue = videos

    async def __anext__(self):
        video = await self._queue.get()
        self.videos = self.videos + [video]
        return video


class YDL:
    """Core controller"""

    def __init__(self, archive: Archive=None, play=False):
        self.archive = archive
        self._aio_loop = asyncio.get_event_loop()
        self.ui = Ui(self, self._aio_loop)

        self.downloads = DownloadManager(aio_loop=self._aio_loop)

        self.videos: List[Video] = []
        self.videos_by_url: Dict[str, Video] = dict()
        if archive:
            for video in archive:
                self.videos.append(video)
                self.videos_by_url[video.url] = video
                self.ui.add_video(video)
                if video.status in {"pending", "downloading"}:
                    noawait(self._handle_new_video(video))

        self._session_added: asyncio.Queue[Video] = asyncio.Queue()

        self._playlists: List[Playlist] = []

    def run(self):
        self._aio_loop.add_signal_handler(signal.SIGINT, self.shutdown)
        log.debug("Starting main loop.")
        self.ui.run_loop()
        log.debug("Waiting for pending tasks to finish…")
        try:
            self.downloads.shutdown()
            while self._playlists:
                self.remove_playlist()
            self._aio_loop.run_until_complete(self._cleanup_tasks())
            self._aio_loop.close()
        except Exception as e:
            log.error(e)
        self.archive.update(self.videos)
        log.debug("Bye.")

    def shutdown(self):
        log.debug("Shutting down.")
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
        if url in self.videos_by_url:
            log.info("Duplicate detected.")
            original = self.videos_by_url[url]
            video.sync_to_original(original)
            await original.finished
            await self._session_added.put(video)
        else:
            self.videos.append(video)
            self.videos_by_url[url] = video
            await self._handle_new_video(video)

    async def _handle_new_video(self, video):
        if video.status == "pending":
            log.debug("Downloading meta info.")
            await self.downloads.download_meta(video)
            if video.status != "error" and os.path.isfile(video.filename):
                # multiple URLs pointing to the same video
                video.status == "duplicate"
        if video.status in {"pending", "downloading"}:
            log.info("Starting download from %s", video.url)
            await self.downloads.download(video)
            log.info("Finished download from %s", video.url)
            await self._session_added.put(video)

    def start_playlist(self):
        self.add_playlist(QueuePlaylist(self._session_added))

    def start_random_playlist(self):
        self.add_playlist(RandomPlaylist(self.videos))

    def add_playlist(self, playlist: Playlist):
        self._playlists.append(playlist)
        self.ui.add_playlist(len(self._playlists) - 1, playlist)
        playlist.start()

    def remove_playlist(self, index: int=None):
        if index is None:
            index = len(self._playlists) - 1
        try:
            playlist = self._playlists.pop(index)
        except IndexError:
            pass
        else:
            playlist.stop()
