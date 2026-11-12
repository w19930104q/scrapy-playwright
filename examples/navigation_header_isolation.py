"""Navigation header isolation example.

Enable PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION (Chromium only) so that the
headers explicitly set on the Scrapy request follow only the initial
main-frame navigation and its same-origin redirect chain.

If that chain ever becomes cross-origin, the Authorization and X-Api-Key
headers are stripped permanently for the rest of the chain, even if a later
redirect returns to the original origin. Origins are compared by scheme, host
and effective port. Cookies keep being managed by the browser.

Run against a site you control that returns cross-origin redirects to observe
the behavior, e.g. with the request inspector on the target origin checking
which headers arrive.
"""

from scrapy import Spider, Request


class NavigationHeaderIsolationSpider(Spider):
    name = "navigation_header_isolation"

    custom_settings = {
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        # Chromium-only: enabling it with firefox/webkit raises NotSupported.
        # A custom PLAYWRIGHT_PROCESS_REQUEST_HEADERS callback (or None) takes
        # precedence and is not affected by this setting.
        "PLAYWRIGHT_NAVIGATION_HEADER_ISOLATION": True,
    }

    def start_requests(self):
        yield Request(
            url="https://example.com/login",
            meta={"playwright": True},
            headers={
                "Authorization": "Bearer secret-token",
                "X-Api-Key": "api-key-123",
                "Custom-Header": "follows-the-whole-chain",
            },
            cookies={"session": "managed-by-the-browser"},
        )

    def parse(self, response, **kwargs):
        yield {"url": response.url, "status": response.status}
