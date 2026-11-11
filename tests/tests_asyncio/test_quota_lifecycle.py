"""Deterministic regressions for page/context quota lifecycle.

These tests do not rely on random sleeps for synchronization: interleavings
are driven explicitly by cancelling tasks or by emitting lifecycle events on
hand-built fakes, and every assertion checks concrete outcomes (live
resources, waiter fate, follow-up crawl success).

They exercise the guarantees required when a browser crawl is closed or
cancelled:

* a cancellation while waiting on a quota, creating a context/page, running
  the init callback or navigating ends as a cancellation and returns every
  quota slot it had taken, so subsequent requests are not starved;
* a page created by the handler is closed when its request is cancelled,
  while a page supplied through ``playwright_page`` is never closed by the
  handler;
* repeated "close"/"crash" events for one page release its slot exactly once;
* after a same-named context is rebuilt, late events from the old object can
  neither release the new quota nor delete the new context;
* closing a context (or losing the browser connection) unblocks requests
  waiting on its page slots: they hit the existing TargetClosedError retry
  path (or fail explicitly) instead of hanging, and crawling resumes.
"""

import asyncio
import logging
import unittest
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

from playwright._impl._errors import TargetClosedError
from playwright.async_api import Error as PlaywrightError
from scrapy import Request, Spider

from scrapy_playwright.page import PageMethod
from tests import make_handler
from tests.mockserver import StaticMockServer


class _FakeEmitter:
    """Minimal synchronous event emitter mimicking the subset of Playwright's
    event API used by the handler (on/remove_listener/event dispatch)."""

    def __init__(self) -> None:
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def once(self, event, callback):
        def wrapper(*args, **kwargs):
            self.remove_listener(event, wrapper)
            callback(*args, **kwargs)

        self.on(event, wrapper)

    def remove_listener(self, event, callback):
        self._listeners.get(event, []).remove(callback)

    def emit(self, event, *args):
        for callback in list(self._listeners.get(event, [])):
            callback(*args)

    def listener_count(self, event):
        return len(self._listeners.get(event, []))


def _make_fake_page():
    page = _FakeEmitter()
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()
    page.new_page = None
    return page


def _make_fake_context(max_pages):
    context = _FakeEmitter()
    context.pages = []
    context.new_page = AsyncMock()
    context.close = AsyncMock()
    context.set_default_navigation_timeout = MagicMock()
    context._semaphore = asyncio.Semaphore(value=max_pages)
    return context


class QuotaLifecycleMixin:
    browser_type = "chromium"

    # ------------------------------------------------------------------
    # event idempotency and same-name rebuild (fakes, no browser needed)
    # ------------------------------------------------------------------

    async def test_duplicate_close_and_crash_events_release_once(self):
        settings = {
            "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
            "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
        }
        async with make_handler(settings) as handler:
            context = _make_fake_context(max_pages=1)
            from scrapy_playwright.handler import BrowserContextWrapper

            wrapper = BrowserContextWrapper(
                context=context,
                semaphore=asyncio.Semaphore(value=1),
                persistent=False,
            )
            handler.context_wrappers["default"] = wrapper

            fake_page = _make_fake_page()
            context.new_page.return_value = fake_page

            request = Request("https://example.org", meta={"playwright": True})
            page = await handler._create_page(request=request, spider=Spider("foo"))
            self.assertEqual(wrapper.semaphore._value, 0)
            self.assertIn(page, wrapper.page_slots)

            # whichever terminal event arrives first releases the single slot
            page.emit("close")
            self.assertEqual(wrapper.semaphore._value, 1)
            self.assertNotIn(page, wrapper.page_slots)
            # duplicate/late events must not over-release
            page.emit("close")
            page.emit("crash")
            self.assertEqual(wrapper.semaphore._value, 1)

            # explicit release after the event is a no-op too
            handler._release_page_slot(page)
            self.assertEqual(wrapper.semaphore._value, 1)

    async def test_explicit_release_is_idempotent_with_supplied_page(self):
        """A page the handler did not create has no slot to release."""
        settings = {"PLAYWRIGHT_BROWSER_TYPE": self.browser_type}
        async with make_handler(settings) as handler:
            context = _make_fake_context(max_pages=1)
            from scrapy_playwright.handler import BrowserContextWrapper

            wrapper = BrowserContextWrapper(
                context=context,
                semaphore=asyncio.Semaphore(value=1),
                persistent=False,
            )
            handler.context_wrappers["default"] = wrapper
            supplied = _make_fake_page()

            handler._release_page_slot(supplied)  # must be a no-op
            self.assertEqual(wrapper.semaphore._value, 1)
            supplied.close.assert_not_called()

    async def test_late_event_of_replaced_same_named_context_is_ignored(self):
        from scrapy_playwright.handler import BrowserContextWrapper

        settings = {
            "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
            "PLAYWRIGHT_MAX_CONTEXTS": 1,
            "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
        }
        async with make_handler(settings) as handler:
            old_context = _make_fake_context(max_pages=1)
            old_wrapper = BrowserContextWrapper(
                context=old_context,
                semaphore=asyncio.Semaphore(value=1),
                persistent=False,
            )
            old_context.on(
                "close",
                handler._make_close_browser_context_callback(
                    name="default",
                    wrapper=old_wrapper,
                    persistent=False,
                    remote=False,
                ),
            )
            handler.context_wrappers["default"] = old_wrapper
            await handler.context_semaphore.acquire()  # the taken context slot

            old_page = _make_fake_page()
            old_context.new_page.return_value = old_page
            request = Request("https://example.org", meta={"playwright": True})
            await handler._create_page(request=request, spider=Spider("foo"))
            self.assertEqual(old_wrapper.semaphore._value, 0)

            # the old context closes: its slot is released and wrapper removed
            old_context.emit("close")
            self.assertNotIn("default", handler.context_wrappers)
            self.assertEqual(handler.context_semaphore._value, 1)
            self.assertTrue(old_wrapper.closed.is_set())

            # a same-named context is rebuilt, taking the freed context slot
            new_context = _make_fake_context(max_pages=1)
            new_wrapper = BrowserContextWrapper(
                context=new_context,
                semaphore=asyncio.Semaphore(value=1),
                persistent=False,
            )
            new_context.on(
                "close",
                handler._make_close_browser_context_callback(
                    name="default",
                    wrapper=new_wrapper,
                    persistent=False,
                    remote=False,
                    spider=Spider("foo"),
                ),
            )
            await handler.context_semaphore.acquire()
            handler.context_wrappers["default"] = new_wrapper

            new_page = _make_fake_page()
            new_context.new_page.return_value = new_page
            await handler._create_page(
                Request("https://example.org/2", meta={"playwright": True}),
                Spider("foo"),
            )
            self.assertEqual(new_wrapper.semaphore._value, 0)

            # late events from the dead objects must not touch new resources
            old_page.emit("close")
            old_page.emit("crash")
            old_context.emit("close")
            self.assertIn("default", handler.context_wrappers)
            self.assertIs(handler.context_wrappers["default"], new_wrapper)
            # the new context still holds its one slot (old events ignored)
            self.assertEqual(handler.context_semaphore._value, 0)
            # the new page still holds its one page slot
            self.assertEqual(new_wrapper.semaphore._value, 0)
            self.assertIn(new_page, new_wrapper.page_slots)

    async def test_context_close_unblocks_page_slot_waiter(self):
        """A waiter blocked on a full context must fail, not hang, when the
        context closes."""
        from scrapy_playwright.handler import BrowserContextWrapper

        settings = {
            "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
            "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
        }
        async with make_handler(settings) as handler:
            context = _make_fake_context(max_pages=1)
            wrapper = BrowserContextWrapper(
                context=context,
                semaphore=asyncio.Semaphore(value=1),
                persistent=False,
            )
            context.on(
                "close",
                handler._make_close_browser_context_callback(
                    name="default",
                    wrapper=wrapper,
                    persistent=False,
                    remote=False,
                ),
            )
            handler.context_wrappers["default"] = wrapper

            # occupy the only slot
            held_page = _make_fake_page()
            context.new_page.return_value = held_page
            await handler._create_page(
                Request("https://example.org", meta={"playwright": True}),
                Spider("foo"),
            )
            self.assertEqual(wrapper.semaphore._value, 0)

            waiter = asyncio.ensure_future(handler._acquire_page_slot(wrapper))
            await asyncio.sleep(0)  # let the waiter block on the semaphore
            self.assertFalse(waiter.done())

            # closing the context settles the waiter with TargetClosedError
            context.emit("close")
            with self.assertRaises(TargetClosedError):
                await asyncio.wait_for(waiter, timeout=1)

            # the failed waiter did not steal or leak a slot
            self.assertEqual(wrapper.semaphore._value, 0)
            waiter2 = asyncio.ensure_future(handler._acquire_page_slot(wrapper))
            with self.assertRaises(TargetClosedError):
                await asyncio.wait_for(waiter2, timeout=1)

    # ------------------------------------------------------------------
    # cancellation interleavings (real browser, real resources)
    # ------------------------------------------------------------------

    async def test_cancel_while_waiting_for_page_slot_returns_quota(self):
        with StaticMockServer() as server:
            settings = {
                "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
                "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
            }
            async with make_handler(settings) as handler:
                url = server.urljoin("/index.html")
                # hold the only page slot for the whole test
                holder = asyncio.ensure_future(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    )
                )
                # wait deterministically until the lazy context exists and
                # the holder owns its only page slot
                while "default" not in handler.context_wrappers:
                    await asyncio.sleep(0)
                wrapper = handler.context_wrappers["default"]
                while wrapper.semaphore._value != 0:
                    await asyncio.sleep(0)
                self.assertEqual(wrapper.semaphore._value, 0)

                blocked = asyncio.ensure_future(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    )
                )
                while not blocked.done() and wrapper.semaphore._value == 0:
                    await asyncio.sleep(0)
                # the blocked request is parked on the slot, nothing was over-acquired
                self.assertFalse(blocked.done())

                blocked.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await blocked
                await holder

                # quota fully returned: the context serves another full request
                self.assertEqual(wrapper.semaphore._value, 1)
                response = await asyncio.wait_for(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    ),
                    timeout=20,
                )
                self.assertEqual(response.status, 200)
                self.assertEqual(wrapper.semaphore._value, 1)

    async def test_cancel_during_navigation_closes_page_and_returns_slot(self):
        with StaticMockServer() as server:
            settings = {
                "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
                "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
            }
            async with make_handler(settings) as handler:
                slow_url = server.urljoin("/index.html?delay=30")

                task = asyncio.ensure_future(
                    handler._download_request(
                        Request(slow_url, meta={"playwright": True}), Spider("foo")
                    )
                )
                # block until the lazy context and its page exist and navigation is in flight
                while "default" not in handler.context_wrappers:
                    await asyncio.sleep(0)
                wrapper = handler.context_wrappers["default"]
                while not wrapper.page_slots:
                    await asyncio.sleep(0)
                page = next(iter(wrapper.page_slots))
                self.assertFalse(page.is_closed())

                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                # the self-created page was closed and its slot returned
                self.assertTrue(page.is_closed())
                self.assertEqual(wrapper.page_slots, set())
                self.assertEqual(wrapper.semaphore._value, 1)

                # follow-up crawl succeeds with the full quota available
                response = await asyncio.wait_for(
                    handler._download_request(
                        Request(server.urljoin("/index.html"), meta={"playwright": True}),
                        Spider("foo"),
                    ),
                    timeout=20,
                )
                self.assertEqual(response.status, 200)
                self.assertEqual(wrapper.semaphore._value, 1)

    async def test_cancel_does_not_close_supplied_page(self):
        with StaticMockServer() as server:
            settings = {"PLAYWRIGHT_BROWSER_TYPE": self.browser_type}
            async with make_handler(settings) as handler:
                setup = Request(
                    server.urljoin("/index.html"),
                    meta={"playwright": True, "playwright_include_page": True},
                )
                setup_response = await handler._download_request(setup, Spider("foo"))
                supplied = setup_response.meta["playwright_page"]
                wrapper = handler.context_wrappers["default"]
                self.assertFalse(supplied.is_closed())
                sem_before = wrapper.semaphore._value  # one slot held by this page

                # a long, reliably-cancellable wait on the caller-supplied page
                slow = Request(
                    server.urljoin("/index.html"),
                    meta={
                        "playwright": True,
                        "playwright_page": supplied,
                        "playwright_include_page": True,
                        "playwright_page_methods": [
                            PageMethod(
                                "wait_for_selector", "#never-appears", timeout=60_000
                            )
                        ],
                    },
                )
                task = asyncio.ensure_future(
                    handler._download_request(slow, Spider("foo"))
                )
                # wait deterministically until the request is parked in the
                # long selector wait; a bounded spin avoids any random timing
                for _ in range(1000):
                    if not task.done():
                        break
                    await asyncio.sleep(0)
                self.assertFalse(task.done())

                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                # caller-owned page survives cancellation: the handler must
                # never close it on its own; ownership (and the matching slot)
                # stays with the caller until the caller closes the page itself
                self.assertFalse(supplied.is_closed())
                # cancellation did not release nor duplicate the held slot
                self.assertEqual(wrapper.semaphore._value, sem_before)
                await supplied.close()
                # closing the caller-owned page now returns its slot via the event
                while wrapper.semaphore._value != sem_before + 1:
                    await asyncio.sleep(0)
                self.assertEqual(wrapper.page_slots, set())

    async def test_new_page_failure_returns_slot_and_context_stays_usable(self):
        with StaticMockServer() as server:
            settings = {
                "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
                "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
            }
            async with make_handler(settings) as handler:
                # bring the lazy context into existence first
                warmup = await handler._download_request(
                    Request(server.urljoin("/index.html"), meta={"playwright": True}),
                    Spider("foo"),
                )
                self.assertEqual(warmup.status, 200)
                wrapper = handler.context_wrappers["default"]
                self.assertEqual(wrapper.semaphore._value, 1)
                with patch.object(
                    wrapper.context,
                    "new_page",
                    new_callable=AsyncMock,
                    side_effect=PlaywrightError("simulated failure"),
                ):
                    with self.assertRaises(PlaywrightError):
                        await handler._create_page(
                            Request(server.urljoin("/index.html"), meta={"playwright": True}),
                            Spider("foo"),
                        )
                self.assertEqual(wrapper.semaphore._value, 1)

                response = await asyncio.wait_for(
                    handler._download_request(
                        Request(server.urljoin("/index.html"), meta={"playwright": True}),
                        Spider("foo"),
                    ),
                    timeout=20,
                )
                self.assertEqual(response.status, 200)
                self.assertEqual(wrapper.semaphore._value, 1)

    async def test_browser_disconnection_unblocks_waiter_and_crawling_resumes(self):
        with StaticMockServer() as server:
            settings = {
                "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
                "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 1,
            }
            async with make_handler(settings) as handler:
                url = server.urljoin("/index.html")

                holder = asyncio.ensure_future(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    )
                )
                while "default" not in handler.context_wrappers:
                    await asyncio.sleep(0)
                wrapper = handler.context_wrappers["default"]
                while wrapper.semaphore._value != 0:
                    await asyncio.sleep(0)

                blocked = asyncio.ensure_future(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    )
                )
                await asyncio.sleep(0)
                self.assertFalse(blocked.done())

                # kill the browser connection while the request is queued.
                # The waiter must settle (retry to completion on the restarted
                # browser, or fail explicitly) instead of hanging forever.
                await handler.browser.close()
                try:
                    blocked_result = await asyncio.wait_for(blocked, timeout=20)
                except (TargetClosedError, PlaywrightError):
                    pass  # explicit failure is an acceptable, non-hanging outcome
                else:
                    self.assertEqual(blocked_result.status, 200)
                with suppress(Exception):
                    await holder

                # the browser is restarted and a fresh request goes through
                response = await asyncio.wait_for(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    ),
                    timeout=20,
                )
                self.assertEqual(response.status, 200)
                new_wrapper = handler.context_wrappers["default"]
                self.assertEqual(new_wrapper.semaphore._value, 1)

    async def test_context_close_releases_context_slot_for_rebuild(self):
        with StaticMockServer() as server:
            settings = {
                "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
                "PLAYWRIGHT_MAX_CONTEXTS": 1,
            }
            async with make_handler(settings) as handler:
                url = server.urljoin("/index.html")
                await handler._download_request(
                    Request(url, meta={"playwright": True}), Spider("foo")
                )
                self.assertEqual(handler.context_semaphore._value, 0)

                await handler.context_wrappers["default"].context.close()
                # deterministic: spin until the close event releases the slot
                while handler.context_semaphore._value != 1:
                    await asyncio.sleep(0)
                self.assertNotIn("default", handler.context_wrappers)

                response = await asyncio.wait_for(
                    handler._download_request(
                        Request(url, meta={"playwright": True}), Spider("foo")
                    ),
                    timeout=20,
                )
                self.assertEqual(response.status, 200)
                self.assertEqual(handler.context_semaphore._value, 0)


class TestQuotaLifecycleChromium(unittest.IsolatedAsyncioTestCase, QuotaLifecycleMixin):
    browser_type = "chromium"


class TestThreadedLoopCancellationSettlement(unittest.IsolatedAsyncioTestCase):
    """A CancelledError raised inside a bridged coroutine must settle the
    returned Future/Deferred instead of escaping and leaving it pending."""

    async def test_cancelled_coro_settles_future(self):
        from scrapy_playwright._loop import _QueueItem, _ThreadedLoopAdapter

        scheduled = []

        class FakeLoop:
            def call_soon_threadsafe(self, callback, *args):
                scheduled.append((callback, args))

        async def cancelled_coro():
            raise asyncio.CancelledError()

        future = asyncio.Future()
        item = _QueueItem(
            coro=cancelled_coro(),
            promise=future,
            loop=FakeLoop(),
        )
        await _ThreadedLoopAdapter._handle_coro_future(item)

        self.assertEqual(len(scheduled), 1)
        _callback, args = scheduled[0]
        self.assertIsInstance(args[0], asyncio.CancelledError)

    async def test_cancelled_coro_settles_deferred(self):
        from scrapy_playwright._loop import _QueueItem, _ThreadedLoopAdapter
        from twisted.internet.defer import Deferred
        from twisted.python.failure import Failure

        results = []

        class FakeReactor:
            def callFromThread(self, callback, *args):
                callback(*args)  # run synchronously, independent of reactor state

        deferred = Deferred()
        deferred.addErrback(lambda failure: results.append(failure))

        async def cancelled_coro():
            raise asyncio.CancelledError()

        item = _QueueItem(coro=cancelled_coro(), promise=deferred)
        # reactor is imported inside the method; patch the shared singleton so
        # callFromThread runs the errback synchronously without a live reactor
        import twisted.internet

        with patch.object(twisted.internet, "reactor", FakeReactor()):
            await _ThreadedLoopAdapter._handle_coro_deferred(item)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Failure)
        self.assertIsInstance(results[0].value, asyncio.CancelledError)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
