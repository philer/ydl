#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import sys

import pyperclip
import urwid
from urwid import (AsyncioEventLoop, AttrMap, Columns, Divider,
                   ExitMainLoop, Filler, Frame, LineBox, ListBox, MainLoop,
                   Overlay, Padding, Pile, SimpleFocusListWalker, Text,
                   WidgetDecoration, WidgetPlaceholder, WidgetWrap)


log = logging.getLogger(__name__)

class VideoWidget(WidgetWrap):
    """Ugly mix of data model and view widget"""

    palette = (
        ("pending",            "light gray",  "", "", "g70",  ""),
        ("duplicate",          "yellow",      "", "", "#da0", ""),
        ("downloading",        "light blue",  "", "", "#6dd", ""),
        ("downloading_filled", "standout",    "", "", "g0",   "#6dd"),
        ("finished",           "",            "", "", "",     ""),
        ("finished_icon",      "light green", "", "", "#8f6", ""),
        ("error",              "light red",   "", "", "#d66", ""),
        ("deleted",            "dark gray,strikethrough", "", "", "#666,strikethrough", ""),
        ("deleted_icon",       "dark gray",   "", "", "#666", ""),
    )
    palette += tuple((p[0] + "_focus", p[1] + ",bold", *p[2:4], p[4] + ",bold", p[5])
                     for p in palette if not p[0].endswith("_icon"))

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
        if status == "downloading" and self._video.progress:
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

    def rows(self, size, focus=False):
        if self._video.status == "deleted" and not self._ui._show_deleted:
            return 0
        return super().rows(size, focus)

    def render(self, size, focus=False):
        """Update appearance based on video status, widget focus and size."""
        if self._video.status == "deleted" and not self._ui._show_deleted:
            return Pile([]).render(size)
        status = self._video.status
        focused = "_focus" if focus else ""
        info = self._info.ljust(size[0])
        if status == "downloading":
            filled_width = int(size[0] * self._video.progress)
            filled, empty = info[:filled_width], info[filled_width:]
            info_text = [("downloading_filled" + focused, filled),
                         ("downloading" + focused, empty)]
        else:
            info_text = (status + focused, info)
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
            except IndexError:
                pass
            return True
        return False


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
        *VideoWidget.palette,
        ("divider",         "dark gray",   "", "", "#666", ""),
        ("divider_focus",   "light gray",  "", "", "#aaa", ""),
        ("button",          "bold",        ""),
        ("button_focus",    "bold,standout",""),
    )

    def __init__(self, core, aio_loop):
        self._core = core
        self._aio_loop = aio_loop

        self._videos = ListBox(SimpleFocusListWalker([]))
        self._root = WidgetPlaceholder(Scrollbar(self._videos))

        self._loop = MainLoop(widget=self._root,
                              palette=self.palette,
                              event_loop=CustomEventLoop(loop=aio_loop),
                              unhandled_input=self._handle_global_input)
        self._loop.screen.set_terminal_properties(
            colors=256, bright_is_bold=False, has_underline=True)

        self._show_deleted = True

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
            self._show_deleted = not self._show_deleted
            for v in self._videos.body:
                v._invalidate()

    def _handle_urls(self, text):
        """Extract valid urls from user input and pass them on."""
        for url in text.split():
            if url:
                self._aio_loop.create_task(self._core.add_video(url))

    def add_video(self, video):
        """Wrap a new video and add it to the display."""
        widget = VideoWidget(self, video)
        self._videos.body.append(widget)
        self._videos.focus_position = len(self._videos.body) - 1

    async def confirm(self, message):
        return "continue" == await Dialog(message, parent=self._root)
