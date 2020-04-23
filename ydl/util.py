import asyncio
from functools import partial
from typing import (cast, Any, Callable, Dict, Iterable, Iterator, List,
                    Optional, Sequence, TypeVar)

noawait = asyncio.create_task


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
