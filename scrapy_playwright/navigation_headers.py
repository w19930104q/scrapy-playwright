"""
Chromium-specific isolation of Scrapy headers along main-frame navigation chains.

When enabled through the ``PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION`` setting, the
explicit headers of the Scrapy request are only injected into the initial
main-frame navigation and the redirect chain that it starts. As soon as the
chain becomes cross-origin, the ``Authorization`` and ``X-Api-Key`` headers are
permanently stripped from the chain, including requests that later go back to
the original origin. Origins are compared by scheme, host and effective port.

Only Chromium is supported: the implementation relies on the raw
Chrome DevTools Protocol (CDP) ``Fetch`` domain, which (unlike Playwright's
routing API) pauses every request in a redirect chain individually and
therefore allows headers to be decided on a per-hop basis.
"""

import asyncio
import base64
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from playwright._impl._errors import Error as PlaywrightError
from playwright.async_api import CDPSession, Page
from scrapy.http.headers import Headers


logger = logging.getLogger("scrapy-playwright")


# Headers that must never leave the origin of the initial navigation.
# Lower-cased for case-insensitive matching against CDP header names.
SENSITIVE_HEADER_NAMES: Tuple[str, ...] = ("authorization", "x-api-key")

# CDP resource type for main-frame and iframe document navigations.
_DOCUMENT_RESOURCE_TYPE = "Document"

# Forbidden request headers, mirroring Playwright's own routing rules (see
# FORBIDDEN_HEADER_NAMES in the Playwright driver). These are managed by the
# browser and cannot be overridden, notably Cookie (the cookie jar) and
# Content-Length (derived from the request body).
_FORBIDDEN_HEADER_NAMES = frozenset(
    {
        "accept-charset",
        "accept-encoding",
        "access-control-request-headers",
        "access-control-request-method",
        "connection",
        "content-length",
        "cookie",
        "date",
        "dnt",
        "expect",
        "host",
        "keep-alive",
        "origin",
        "referer",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "via",
    }
)


def _is_forbidden_header(name: str) -> bool:
    """Replicate Playwright's forbidden-header rule for routing overrides."""
    lowered = name.lower()
    return (
        lowered in _FORBIDDEN_HEADER_NAMES
        or lowered.startswith("proxy-")
        or lowered.startswith("sec-")
    )


def _origin_tuple(url: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Return the origin of a URL as a (scheme, host, effective port) tuple."""
    parsed = urlparse(url)
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    elif parsed.scheme == "http":
        port = 80
    else:
        port = None
    return parsed.scheme, parsed.hostname, port


def _headers_to_cdp(headers: Dict[str, str]) -> List[dict]:
    return [{"name": name, "value": value} for name, value in headers.items()]


def _set_header_case_insensitive(headers: Dict[str, str], name: str, value: str) -> None:
    """Set a header, replacing any existing entry regardless of its casing.

    CDP header names are case-insensitive and Chromium normalizes them, but
    keeping a single canonical entry avoids relying on ordering when the same
    header is present under different casings.
    """
    for existing in list(headers):
        if existing.lower() == name.lower():
            del headers[existing]
    headers[name] = value


@dataclass
class _NavigationChain:
    """State of a single main-frame navigation, keyed by its CDP network id."""

    origin: Tuple[str, Optional[str], Optional[int]]
    # all overridable headers explicitly set on the Scrapy request, lower-cased
    explicit_headers: Dict[str, str]
    # becomes True forever once the chain goes cross-origin
    stripped: bool = False


class NavigationHeaderIsolation:
    """Per-page CDP interceptor implementing navigation-chain header isolation.

    One instance is attached to each page while the feature is enabled. It only
    intercepts document requests:

    - the *initial* main-frame document (the Scrapy navigation) and every hop of
      its redirect chain receive the explicit Scrapy headers, subject to
      cross-origin stripping;
    - later, independent main-frame navigations (e.g. triggered by PageMethods)
      and iframe documents keep the browser-managed headers and only inherit the
      Scrapy User-Agent;
    - subresources are not intercepted at all (Document-only pattern).

    Forbidden headers such as Cookie or Content-Length are never overridden and
    remain managed by the browser.
    """

    def __init__(
        self,
        page: Page,
        scrapy_headers: Headers,
        scrapy_method: str,
        scrapy_body: Optional[bytes],
    ) -> None:
        self.page = page
        self.scrapy_method = scrapy_method.upper()
        self.scrapy_body = scrapy_body
        # shared with the download handler, mutated in place on the initial hop
        self.scrapy_headers = scrapy_headers
        # headers explicitly defined on the Scrapy request; multiple values for
        # one name are comma-joined, matching the former to_unicode_dict() output
        scrapy_headers_str: Dict[str, str] = {}
        for name, value in scrapy_headers.to_tuple_list():
            if name in scrapy_headers_str:
                scrapy_headers_str[name] = f"{scrapy_headers_str[name]},{value}"
            else:
                scrapy_headers_str[name] = value
        # lower-cased keys, excluding headers the browser must keep managing itself
        self.explicit_headers: Dict[str, str] = {
            name.lower(): value
            for name, value in scrapy_headers_str.items()
            if not _is_forbidden_header(name)
        }
        self.scrapy_user_agent: Optional[str] = self.explicit_headers.get("user-agent")
        self.chains: Dict[str, _NavigationChain] = {}
        # only the first main-frame document belongs to the Scrapy navigation;
        # subsequent documents with a new network id are independent navigations
        self.initial_navigation_seen = False
        self._cdp_session: Optional[CDPSession] = None
        self._main_frame_id: Optional[str] = None
        self._closed = False

    async def start(self) -> None:
        """Enable CDP interception and register the request-paused listener."""
        context = self.page.context
        self._cdp_session = await context.new_cdp_session(self.page)
        frame_tree = await self._cdp_session.send("Page.getFrameTree")
        self._main_frame_id = frame_tree["frameTree"]["frame"]["id"]
        self._cdp_session.on("Fetch.requestPaused", self._schedule_handling)
        # Document only: subresources (images, scripts, ...) are not intercepted.
        await self._cdp_session.send(
            "Fetch.enable",
            {"patterns": [{"urlPattern": "*", "resourceType": _DOCUMENT_RESOURCE_TYPE}]},
        )

    def _schedule_handling(self, event: dict) -> None:
        # CDP event listeners are invoked synchronously; schedule the actual
        # handling as a task so the event dispatcher is not blocked and the
        # coroutine's exceptions are not silently swallowed.
        if not self._closed:
            asyncio.create_task(self._handle_request_paused(event))

    @property
    def _cdp(self) -> CDPSession:
        """The active CDP session, only available between start() and close()."""
        if self._cdp_session is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Navigation header interception has not been started")
        return self._cdp_session

    async def _handle_request_paused(self, event: dict) -> None:
        request_id = event["requestId"]
        request = event["request"]
        url = request["url"]
        try:
            is_main_frame = event.get("frameId") == self._main_frame_id
            network_id = event["networkId"]
            chain = self.chains.get(network_id)

            if not is_main_frame:
                # iframe document: browser headers, only the User-Agent carries over
                await self._continue_with_user_agent(request_id, request)
                return

            if chain is not None:
                # a further hop of the tracked Scrapy navigation chain
                await self._handle_redirect_hop(request_id, request, url, chain)
            elif not self.initial_navigation_seen:
                # the initial main-frame document of this Scrapy request
                await self._handle_initial_hop(request_id, request, url, network_id)
            else:
                # an independent main-frame navigation (e.g. a PageMethod):
                # browser headers, only the User-Agent carries over
                await self._continue_with_user_agent(request_id, request)
        except Exception as ex:  # pragma: no cover - defensive, never block navigation
            logger.warning(
                "Navigation header isolation failed for %s: %s; continuing request",
                url,
                ex,
                exc_info=True,
            )
            with suppress(PlaywrightError):
                await self._cdp.send("Fetch.continueRequest", {"requestId": request_id})

    async def _continue_with_user_agent(self, request_id: str, request: dict) -> None:
        """Continue a document that is not part of the Scrapy chain, applying
        only the Scrapy User-Agent override to the browser-managed headers."""
        headers = dict(request["headers"])
        if self.scrapy_user_agent:
            _set_header_case_insensitive(headers, "user-agent", self.scrapy_user_agent)
        await self._cdp.send(
            "Fetch.continueRequest",
            {"requestId": request_id, "headers": _headers_to_cdp(headers)},
        )

    async def _handle_initial_hop(
        self, request_id: str, request: dict, url: str, network_id: str
    ) -> None:
        """First request of a navigation: inject all overridable explicit Scrapy
        headers, plus the Scrapy method and body when applicable."""
        # start from the browser's headers (retaining forbidden ones), then layer
        # the Scrapy headers on top, skipping those the browser must keep managing
        headers = dict(request["headers"])
        for name, value in self.explicit_headers.items():
            _set_header_case_insensitive(headers, name, value)

        overrides: dict = {
            "requestId": request_id,
            "headers": _headers_to_cdp(headers),
        }
        if self.scrapy_method != request.get("method", "GET").upper():
            overrides["method"] = self.scrapy_method
        if self.scrapy_body:
            overrides["postData"] = base64.b64encode(self.scrapy_body).decode("ascii")

        await self._cdp.send("Fetch.continueRequest", overrides)
        self.initial_navigation_seen = True
        self.chains[network_id] = _NavigationChain(
            origin=_origin_tuple(url),
            explicit_headers=dict(self.explicit_headers),
        )
        # mirror the Scrapy-level headers back into the request object, the same
        # way the regular request handler does; browser-managed headers are not
        # copied into the Scrapy request
        self.scrapy_headers.clear()
        self.scrapy_headers.update(Headers(dict(self.explicit_headers)))

    async def _handle_redirect_hop(
        self,
        request_id: str,
        request: dict,
        url: str,
        chain: _NavigationChain,
    ) -> None:
        """Subsequent request of an already tracked navigation chain."""
        headers = dict(request["headers"])
        if chain.stripped or _origin_tuple(url) != chain.origin:
            # cross-origin, now or earlier in the chain: the sensitive headers are
            # gone for good, even if a later hop goes back to the original origin
            chain.stripped = True
            headers = {
                name: value
                for name, value in headers.items()
                if name.lower() not in SENSITIVE_HEADER_NAMES
            }
            for name, value in chain.explicit_headers.items():
                if name.lower() not in SENSITIVE_HEADER_NAMES:
                    _set_header_case_insensitive(headers, name, value)
        else:
            # same-origin hop: Chromium may drop some headers on redirects
            # (e.g. Authorization), re-inject every explicit header of the chain
            for name, value in chain.explicit_headers.items():
                _set_header_case_insensitive(headers, name, value)

        await self._cdp.send(
            "Fetch.continueRequest",
            {"requestId": request_id, "headers": _headers_to_cdp(headers)},
        )

    async def close(self) -> None:
        """Disable interception and detach the CDP session."""
        if self._closed:
            return
        self._closed = True
        self.chains.clear()
        if self._cdp_session is not None:
            with suppress(PlaywrightError):
                await self._cdp_session.send("Fetch.disable")
            with suppress(PlaywrightError):
                await self._cdp_session.detach()
            self._cdp_session = None
