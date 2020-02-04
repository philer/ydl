#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import sys

import pyperclip
import urwid
from urwid import (AsyncioEventLoop, AttrMap, Columns, Divider, ExitMainLoop,
                   Filler, LineBox, ListBox, MainLoop, Overlay, Padding, Pile,
                   SimpleFocusListWalker, Text,
                   WidgetDecoration, WidgetPlaceholder, WidgetWrap)


log = logging.getLogger(__name__)

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
    focus_palette = []
    for style in palette:
        name, fg, bg, mono, fg256, bg256 = style
        if not name.endswith("_icon"):
            focus_palette.append((name + "_focus", fg + ",bold", bg, mono,
                            fg256 + ",bold", bg256 or "g19"))
    palette += focus_palette
    del focus_palette

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
            return f"{self._video.extractor}:{self._video.id} - {self._video.title}"
        return self._video.url

    def __init__(self, ui, video):
        self._ui = ui
        self._video = video

        self._status_widget = Text(" ? ")
        self.update_status_icon()
        self._info_widget = Text("", wrap='clip')
        columns = [
            (3, self._status_widget),
            self._info_widget,
        ]
        self._root = Columns(columns, dividechars=1)
        super().__init__(self._root)

        video.observers.append(self._handle_update)

    def _handle_update(self, _video, prop, _value):
        if prop in {"status", "progress"}:
            self.update_status_icon()
        self._invalidate()

    def update_status_icon(self):
        status = self._video.status
        style = status + "_icon" if status in {"finished", "deleted"} else status
        if status == "downloading" and 0 < self._video.progress < 1:
            icon = f"{self._video.progress: >3.0%}"
        else:
            icon = self.status_icon[status]
        self._status_widget.set_text((style, icon))

    async def _delete(self):
        if await self._ui.confirm("Really delete?"):
            self._video.delete()

    async def _re_download(self):
        if (self._video.status in {"deleted", "error"}
            and await self._ui.confirm("Download this video again?")):
            self._video.status = "pending"

            # TODO This feels wrong.
            await self._ui._core._handle_new_video(self._video)

    def selectable(self):
        return True

    def keypress(self, _size, key):
        if key == "enter" or key == "right" or key == " ":
            if self._video.status in ("error", "deleted"):
                self._ui._aio_loop.create_task(self._re_download())
            else:
                self._ui._aio_loop.create_task(self._video.play())
        elif key == "d" or key == "delete":
            self._ui._aio_loop.create_task(self._delete())
        else:
            return key

    def render(self, size, focus=False):
        """Update appearance based on video status, widget focus and size."""
        width = size[0]
        status = self._video.status
        focused = "_focus" if focus else ""
        if status == "downloading":
            info = f"{self._info:<{width}.{width}}"
            filled_width = int(width * self._video.progress)
            filled, empty = info[:filled_width], info[filled_width:]
            info_text = [("downloading_filled" + focused, filled),
                         ("downloading" + focused, empty)]
        else:
            info_text = (status + focused, self._info)
        self._info_widget.set_text(info_text)
        return self._root.render(size, focus)


class Scrollbar(WidgetDecoration, WidgetWrap):
    """Wrap a ListBox to make it scrollable. This is a box widget."""

    _sizing = frozenset(["box"])

    def __init__(self, list_box: ListBox):
        self._list_box = list_box
        self._bar = Text("")
        root = Columns([list_box, (1, Filler(self._bar))])
        WidgetDecoration.__init__(self, list_box)
        WidgetWrap.__init__(self, root)

    def render(self, size, focus=False):
        self._render_scrollbar(size, focus)
        return WidgetWrap.render(self, size, focus)

    def _render_scrollbar(self, size, focus):
        maxcol, maxrow = size
        size = maxcol - 1, maxrow

        # necessary side effect: calculate correct focus offset/inset
        self._list_box.calculate_visible(size, focus)

        focus_widget, _ = self._list_box.get_focus()
        if not focus_widget:
            self._bar.set_text("│" * maxrow)
            return

        focus_offset, focus_inset = self._list_box.get_focus_offset_inset(size)
        total_height, height_above = 0, 0
        for w in self._list_box.body:
            height = w.rows(size=(maxcol,))
            if w is focus_widget:
                height_above = total_height - focus_offset + focus_inset
            total_height += height

        # get integer heights. using special box drawing characters (╽╿│┃)
        # we get twice the resolution
        visible = int(2 * maxrow * min(1, maxrow / total_height) + 0.5)
        above = int(2 * maxrow * height_above / total_height + 0.5)
        below = 2 * maxrow - above - visible

        top_full, top_half = divmod(above, 2)
        if top_half: visible -= 1
        middle_full, middle_half = divmod(visible, 2)
        if middle_half: below -= 1
        bottom_full, bottom_half = divmod(below, 2)
        top = "│" * top_full + "╽" * top_half
        middle = "┃" * middle_full + "╿" * middle_half
        bottom = "╿" * bottom_half + "│" * bottom_full
        self._bar.set_text(top + middle + bottom)

    def scroll_to(self, fraction):
        """Scroll the ListBox to center the appropriate widget."""
        self._list_box.focus_position = int(fraction * len(self._list_box.body))

    def mouse_event(self, size, event, button, col, row, focus):
        maxcol, maxrow = size
        if col == maxcol - 1:
            if button == 1 and event in {'mouse press', 'mouse drag'}:
                self.scroll_to(row / maxrow)
                return True
            return False
        if self._list_box.mouse_event(size, event, button, col, row, focus):
            return True
        if 4 <= button <= 5:  # scrollwheel
            try:
                self._list_box.focus_position += 1 if button == 5 else -1
                self._invalidate()
            except IndexError:
                pass
            return True
        return False


class VideoList(Scrollbar):

    def __init__(self, videos=[]):
        super().__init__(ListBox(SimpleFocusListWalker(videos)))

    def set_videos(self, videos):
        videos = list(videos)
        old_videos = self._list_box.body
        focus = self._list_box.focus_position
        self._list_box.body = SimpleFocusListWalker(videos)

        focus_candidates = old_videos[focus:] + old_videos[focus - 1::-1]
        lookup = {vid: idx for idx, vid in enumerate(videos)}
        for vid in focus_candidates:
            try:
                focus = lookup[vid]
                break
            except KeyError:
                pass
        else:
            return
        self._list_box.focus_position = focus

    def remove(self, video):
        focus = self._list_box.focus_position
        self._list_box.body.remove(video)
        try:
            self._list_box.focus_position = focus
        except IndexError:
            self._list_box.focus_position = focus - 1

    def append(self, video):
        self._list_box.body.append(video)
        self._list_box.focus_position = len(self._list_box.body) - 1



class LogHandlerWidget(WidgetWrap, logging.Handler):
    """Show log messages in a scrollable window for live debugging."""

    palette = (
        ("log_DEBUG", "",  ""),
        ("log_INFO", "light blue",  ""),
        ("log_WARNING", "yellow", ""),
        ("log_ERROR", "light red", ""),
        ("log_EXCEPTION", "log_ERROR"),
        ("log_CRITICAL", "log_ERROR"),
        ("log_NOTSET", "log_DEBUG"),
    )

    def __init__(self, logger=None, level=logging.DEBUG):
        self._records = ListBox(SimpleFocusListWalker([]))
        WidgetWrap.__init__(self, Scrollbar(self._records))

        logging.Handler.__init__(self)
        self.setLevel(level)
        (logger or logging.getLogger(__package__)).addHandler(self)

    def emit(self, record):
        style = f"log_{record.levelname}"
        text = f"{record.levelname} {record.name}: {record.getMessage()}"
        self._records.body.append(Text((style, text)))
        self._records.focus_position = len(self._records.body) - 1

class Button(WidgetWrap):
    """Custom button with a simpler style."""

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
        try:
            exc = context["exception"]
        except KeyError:
            exc = RuntimeError(context["message"])
        loop.stop()
        if not isinstance(exc, ExitMainLoop):
            log.exception(exc)
            # Store the exc_info so we can re-raise after the loop stops
            self._exc_info = sys.exc_info()
            if self._exc_info == (None, None, None):
                self._exc_info = (type(exc), exc, exc.__traceback__)


class Ui:
    palette = (
        # (name, foreground, background, mono, foreground_high, background_high)
        *VideoWidget.palette,
        *LogHandlerWidget.palette,
        ("divider",         "dark gray",   "", "", "#666", ""),
        ("divider_focus",   "light gray",  "", "", "#aaa", ""),
        ("button",          "bold",        ""),
        ("button_focus",    "bold,standout",""),
    )

    def __init__(self, core, aio_loop):
        self._core = core
        self._aio_loop = aio_loop

        self._show_deleted = False

        self._videos_to_widgets = dict()
        self._video_list = VideoList()
        self._visible_windows = Pile([self._video_list])
        self._root = WidgetPlaceholder(self._visible_windows)

        self._loop = MainLoop(widget=self._root,
                              palette=self.palette,
                              event_loop=CustomEventLoop(loop=aio_loop),
                              unhandled_input=self._handle_global_input)
        self._loop.screen.set_terminal_properties(
            colors=256, bright_is_bold=False, has_underline=True)

        self._log_window = Pile([AttrMap(Divider("─"), "divider"), (5, LogHandlerWidget())]), ('pack', None)
        self._log_window_visible = False

    def run_loop(self):
        self._loop.run()

    def halt_loop(self, *_):
        raise ExitMainLoop()

    def _handle_global_input(self, key):
        if key == "esc" or key == "q":
            self._core.shutdown()
        elif key == "ctrl v":
            try:
                self._handle_urls(pyperclip.paste())
            except pyperclip.PyperclipException:
                pass
        elif key == "p":
            self._core.start_playlist()
        elif key == "P":
            self._core.stop_playlist()
        elif key == "r":
            self._core.start_random_playlist()
        elif key == "R":
            self._core.stop_random_playlist()
        elif key == "h":
            self.toggle_show_deleted()
        elif key == "l":
            self.toggle_log_window()

    def _handle_urls(self, text):
        """Extract valid urls from user input and pass them on."""
        for url in text.split():
            if url:
                self._aio_loop.create_task(self._core.add_video(url))

    def add_video(self, video):
        """Wrap a new video and add it to the display."""
        vw = VideoWidget(self, video)
        self._videos_to_widgets[video] = vw
        if self._show_deleted or video.status != "deleted":
            self._video_list.append(vw)
        video.observers.append(self._video_updated)

    def _video_updated(self, video, attribute, value):
        if value == "deleted" and attribute == "status" and not self._show_deleted:
            self._video_list.remove(self._videos_to_widgets[video])

    def toggle_show_deleted(self):
        self._show_deleted = not self._show_deleted
        if self._show_deleted:
            show = self._videos_to_widgets.values()
        else:
            show = [vw for v, vw in self._videos_to_widgets.items()
                       if v.status != "deleted"]
        self._video_list.set_videos(show)

    def toggle_log_window(self):
        self._log_window_visible = not self._log_window_visible
        if self._log_window_visible:
            self._visible_windows.contents.append(self._log_window)
        else:
            self._visible_windows.contents.remove(self._log_window)

    async def confirm(self, message):
        return "continue" == await Dialog(message, parent=self._root)
