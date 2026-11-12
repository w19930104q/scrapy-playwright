"""Tests for the PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION setting.

These tests use two local HTTP servers (two origins) and inspect the headers,
method and body actually received on every hop of the navigation chains.
Only Chromium supports the feature, which relies on the CDP Fetch domain.
"""

import json
from unittest import IsolatedAsyncioTestCase
from urllib.parse import quote

import pytest
from scrapy import FormRequest, Request, Spider
from scrapy.exceptions import NotSupported
from scrapy.settings import Settings

from scrapy_playwright.handler import Config
from scrapy_playwright.navigation_headers import _origin_tuple
from scrapy_playwright.page import PageMethod

from tests import allow_windows, make_handler, BaseTestCase
from tests.header_isolation_server import HeaderIsolationMockServer


SENSITIVE = ("authorization", "x-api-key")
EXPLICIT_HEADERS = {
    "Authorization": "Bearer secret-token",
    "X-Api-Key": "api-key-123",
    "Custom-Header": "custom-value",
}


class TestOriginComparison(IsolatedAsyncioTestCase):
    def test_origin_tuple_scheme_host_port(self):
        # same origin
        assert _origin_tuple("http://example.com/a") == _origin_tuple("http://example.com/b")
        assert _origin_tuple("https://example.com/a") == (
            "https",
            "example.com",
            443,
        )
        assert _origin_tuple("http://example.com:80/a") == _origin_tuple("http://example.com/b")
        # different scheme
        assert _origin_tuple("http://example.com/a") != _origin_tuple("https://example.com/a")
        # different host
        assert _origin_tuple("http://a.example.com/") != _origin_tuple("http://b.example.com/")
        # different explicit port / effective port
        assert _origin_tuple("http://example.com:8080/") != _origin_tuple("http://example.com/")
        assert _origin_tuple("https://example.com:443/") == _origin_tuple("https://example.com/")


class TestConfigValidation(IsolatedAsyncioTestCase):
    def test_default_disabled(self):
        config = Config.from_settings(Settings({}))
        assert config.navigation_header_isolation is False

    def test_chromium_allowed(self):
        config = Config.from_settings(Settings({"PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION": True}))
        assert config.navigation_header_isolation is True

    def test_other_browsers_rejected(self):
        for browser_type in ("firefox", "webkit"):
            with pytest.raises(NotSupported) as exc_info:
                Config.from_settings(
                    Settings(
                        {
                            "PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION": True,
                            "PLAYWRIGHT_BROWSER_TYPE": browser_type,
                        }
                    )
                )
            assert "only supported" in str(exc_info.value)
            assert browser_type in str(exc_info.value)


class MixinNavigationHeaderIsolationTestCase(BaseTestCase):
    browser_type: str

    @pytest.fixture(autouse=True)
    def setup_servers(self):
        # two distinct origins: different ports on 127.0.0.1
        with HeaderIsolationMockServer() as server_a, HeaderIsolationMockServer() as server_b:
            self.server_a = server_a
            self.server_b = server_b
            yield

    def _settings(self, extra=None):
        settings = {
            "PLAYWRIGHT_BROWSER_TYPE": self.browser_type,
            "PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION": True,
        }
        if extra:
            settings.update(extra)
        return settings

    @staticmethod
    def _lower_headers(record):
        return {key.lower(): value for key, value in record["headers"].items()}

    @allow_windows
    async def test_same_origin_redirect_keeps_sensitive_headers(self):
        """Sensitive headers follow the whole same-origin redirect chain."""
        async with make_handler(self._settings()) as handler:
            url = self.server_a.urljoin(
                f"/redirect?status=307&to={self.server_a.urljoin('/final')}"
            )
            req = Request(url, meta={"playwright": True}, headers=dict(EXPLICIT_HEADERS))
            await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        assert len(records) == 2
        for record in records:
            received = self._lower_headers(record)
            assert received["authorization"] == "Bearer secret-token"
            assert received["x-api-key"] == "api-key-123"
            assert received["custom-header"] == "custom-value"
        # the Scrapy request keeps the headers that were actually sent
        assert req.headers["Authorization"].decode() == "Bearer secret-token"
        # browser-managed headers must not leak into the Scrapy request object
        request_header_names = {name.decode().lower() for name in req.headers}
        assert "host" not in request_header_names
        assert "connection" not in request_header_names
        assert request_header_names == {name.lower() for name in EXPLICIT_HEADERS}

    @allow_windows
    async def test_cross_origin_strips_sensitive_headers_permanently(self):
        """Once cross-origin, sensitive headers are stripped even when going back."""
        async with make_handler(self._settings()) as handler:
            # chain: origin A -> origin B -> back to origin A.
            # Nested Location values must be percent-encoded so the inner query
            # string is not parsed as part of the outer request.
            back_url = quote(self.server_a.urljoin("/final"), safe="")
            cross_url = quote(
                self.server_b.urljoin(f"/redirect?status=302&to={back_url}"),
                safe="",
            )
            url = self.server_a.urljoin(f"/redirect?status=302&to={cross_url}")
            req = Request(url, meta={"playwright": True}, headers=dict(EXPLICIT_HEADERS))
            await handler._download_request(req, Spider("foo"))

        records_a = self.server_a.fetch_records()
        records_b = self.server_b.fetch_records()

        # first hop on the original origin still carries everything
        first = self._lower_headers(records_a[0])
        assert first["authorization"] == "Bearer secret-token"
        assert first["x-api-key"] == "api-key-123"

        # cross-origin hop: sensitive headers gone, non-sensitive ones remain
        cross = self._lower_headers(records_b[0])
        assert "authorization" not in cross
        assert "x-api-key" not in cross
        assert cross["custom-header"] == "custom-value"

        # back on the original origin: still stripped (permanent)
        back = self._lower_headers(records_a[-1])
        assert "authorization" not in back
        assert "x-api-key" not in back
        assert back["custom-header"] == "custom-value"

    @allow_windows
    async def test_303_turns_post_into_get_and_drops_body(self):
        """303 See Other: the browser switches to GET and the body is removed."""
        async with make_handler(self._settings()) as handler:
            url = self.server_a.urljoin(
                f"/redirect?status=303&to={self.server_a.urljoin('/final')}"
            )
            req = FormRequest(
                url,
                meta={"playwright": True},
                formdata={"field": "value303"},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        assert [record["method"] for record in records] == ["POST", "GET"]
        assert records[0]["body"] == "field=value303"
        assert records[1]["body"] == ""
        assert "Content-Length" not in self._lower_headers(records[1])
        # same-origin chain keeps the sensitive headers on both hops
        for record in records:
            received = self._lower_headers(record)
            assert received["authorization"] == "Bearer secret-token"

    @allow_windows
    async def test_307_preserves_method_and_body(self):
        """307 Temporary Redirect: method and body are preserved on the chain."""
        async with make_handler(self._settings()) as handler:
            url = self.server_a.urljoin(
                f"/redirect?status=307&to={self.server_a.urljoin('/final')}"
            )
            req = FormRequest(
                url,
                meta={"playwright": True},
                formdata={"field": "value307"},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        assert [record["method"] for record in records] == ["POST", "POST"]
        assert all(record["body"] == "field=value307" for record in records)
        for record in records:
            received = self._lower_headers(record)
            assert received["authorization"] == "Bearer secret-token"
            assert received["x-api-key"] == "api-key-123"

    @allow_windows
    async def test_308_preserves_method_and_body(self):
        """308 Permanent Redirect: method and body are preserved on the chain."""
        async with make_handler(self._settings()) as handler:
            url = self.server_a.urljoin(
                f"/redirect?status=308&to={self.server_a.urljoin('/final')}"
            )
            req = FormRequest(
                url,
                meta={"playwright": True},
                formdata={"field": "value308"},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        assert [record["method"] for record in records] == ["POST", "POST"]
        assert all(record["body"] == "field=value308" for record in records)

    @allow_windows
    async def test_form_navigation_headers(self):
        """A direct form POST (no redirect) sends the explicit headers and body."""
        async with make_handler(self._settings()) as handler:
            req = FormRequest(
                self.server_a.urljoin("/form"),
                meta={"playwright": True},
                formdata={"username": "alice", "password": "secret"},
                headers=dict(EXPLICIT_HEADERS),
            )
            response = await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        assert len(records) == 1
        record = records[0]
        assert record["method"] == "POST"
        assert record["body"] == "username=alice&password=secret"
        received = self._lower_headers(record)
        assert received["authorization"] == "Bearer secret-token"
        assert received["x-api-key"] == "api-key-123"
        payload = json.loads(response.css("pre::text").get())
        assert payload["method"] == "POST"
        assert payload["body"] == "username=alice&password=secret"

    @allow_windows
    async def test_page_reuse_does_not_leak_previous_headers(self):
        """Reusing a page must not carry the previous request's headers."""
        async with make_handler(self._settings()) as handler:
            spider = Spider("foo")
            # first navigation with sensitive headers, keep the page open
            req1 = Request(
                self.server_a.urljoin("/headers"),
                meta={"playwright": True, "playwright_include_page": True},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req1, spider)
            page = req1.meta["playwright_page"]

            # second navigation on the same page, without sensitive headers
            req2 = Request(
                self.server_b.urljoin("/headers"),
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page": page,
                },
                headers={"Custom-Header": "only-this-one"},
            )
            await handler._download_request(req2, spider)
            await page.close()

        records_a = self.server_a.fetch_records()
        records_b = self.server_b.fetch_records()
        first = self._lower_headers(records_a[0])
        assert first["authorization"] == "Bearer secret-token"
        second = self._lower_headers(records_b[0])
        assert "authorization" not in second
        assert "x-api-key" not in second
        assert second["custom-header"] == "only-this-one"

    @allow_windows
    async def test_pagemethod_navigation_uses_browser_headers(self):
        """A navigation triggered by a PageMethod is not part of the Scrapy
        chain: it uses browser headers, only the User-Agent override carries over."""
        async with make_handler(self._settings()) as handler:
            req = Request(
                self.server_a.urljoin("/headers"),
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod(
                            "goto",
                            self.server_b.urljoin("/headers"),
                        )
                    ],
                },
                headers={**dict(EXPLICIT_HEADERS), "User-Agent": "scrapy-ua-test"},
            )
            await handler._download_request(req, Spider("foo"))

        records_a = self.server_a.fetch_records()
        records_b = self.server_b.fetch_records()
        # initial Scrapy navigation gets the explicit headers
        first = self._lower_headers(records_a[0])
        assert first["authorization"] == "Bearer secret-token"
        assert first["user-agent"] == "scrapy-ua-test"
        # PageMethod-triggered navigation: no Scrapy sensitive headers,
        # but the User-Agent override still applies
        second = self._lower_headers(records_b[0])
        assert "authorization" not in second
        assert "x-api-key" not in second
        assert second["user-agent"] == "scrapy-ua-test"

    @allow_windows
    async def test_iframe_not_affected_by_isolation(self):
        """Iframe documents are outside the main-frame chain and keep browser
        headers (no Scrapy sensitive header injection)."""
        iframe_url = self.server_b.urljoin("/headers")
        async with make_handler(self._settings()) as handler:
            req = Request(
                self.server_a.urljoin("/headers"),
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod(
                            "evaluate",
                            f"document.body.innerHTML ="
                            f" '<iframe src=\"{iframe_url}\"></iframe>';"
                            " new Promise(r => setTimeout(r, 800));",
                        )
                    ],
                },
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records_b = self.server_b.fetch_records()
        iframe_record = next(record for record in records_b if "/headers" in record["path"])
        received = self._lower_headers(iframe_record)
        assert "authorization" not in received
        assert "x-api-key" not in received

    @allow_windows
    async def test_cookies_managed_by_browser(self):
        """Cookies remain managed by the browser. Cookies seeded into the browser
        context are sent automatically, while an explicit Cookie header from the
        Scrapy request is a forbidden header and cannot override the jar."""
        origin = self.server_a.origin
        async with make_handler(self._settings()) as handler:
            req = Request(
                self.server_a.urljoin("/headers"),
                meta={
                    "playwright": True,
                    "playwright_context_kwargs": {
                        "storage_state": {
                            "cookies": [
                                {
                                    "name": "session",
                                    "value": "browser-cookie",
                                    "url": origin,
                                }
                            ]
                        }
                    },
                },
                # an explicit Cookie header must be ignored, the jar wins
                headers={**dict(EXPLICIT_HEADERS), "Cookie": "tampered=ignored"},
            )
            response = await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        received = self._lower_headers(records[0])
        # the browser-managed cookie is sent, the attempted override is not
        assert "session=browser-cookie" in received.get("cookie", "")
        assert "tampered" not in received.get("cookie", "")
        # sanity: the response itself was processed normally
        assert response.status == 200

    @allow_windows
    async def test_none_header_setting_takes_full_control(self):
        """PLAYWRIGHT_PROCESS_REQUEST_HEADERS=None disables Scrapy header
        injection entirely, even with isolation enabled: no extra filtering."""
        settings = self._settings({"PLAYWRIGHT_PROCESS_REQUEST_HEADERS": None})
        async with make_handler(settings) as handler:
            req = Request(
                self.server_a.urljoin("/headers"),
                meta={"playwright": True},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records = self.server_a.fetch_records()
        received = self._lower_headers(records[0])
        # the browser strategy wins: Scrapy explicit headers are not injected
        assert "authorization" not in received
        assert "x-api-key" not in received
        assert "custom-header" not in received

    @allow_windows
    async def test_custom_header_callback_takes_full_control(self):
        """A custom PLAYWRIGHT_PROCESS_REQUEST_HEADERS callback is applied as-is
        without the isolation layer adding or filtering any header."""

        async def custom_headers(
            *,
            browser_type_name,  # pylint: disable=unused-argument
            playwright_request,  # pylint: disable=unused-argument
            scrapy_request_data,  # pylint: disable=unused-argument
        ) -> dict:
            # the callback dictates the full strategy, including a sensitive
            # header that isolation would otherwise strip cross-origin
            return {"foo": "bar", "authorization": "callback-token"}

        settings = self._settings({"PLAYWRIGHT_PROCESS_REQUEST_HEADERS": custom_headers})
        cross_url = quote(self.server_b.urljoin("/headers"), safe="")
        url = self.server_a.urljoin(f"/redirect?status=302&to={cross_url}")
        async with make_handler(settings) as handler:
            req = Request(
                url,
                meta={"playwright": True},
                headers=dict(EXPLICIT_HEADERS),
            )
            await handler._download_request(req, Spider("foo"))

        records_b = self.server_b.fetch_records()
        received = self._lower_headers(records_b[-1])
        # the callback's headers survive untouched, even across origins
        assert received["foo"] == "bar"
        assert received["authorization"] == "callback-token"
        # headers not returned by the callback are not injected by isolation
        assert "x-api-key" not in received


class TestNavigationHeaderIsolationChromium(
    IsolatedAsyncioTestCase, MixinNavigationHeaderIsolationTestCase
):
    browser_type = "chromium"
