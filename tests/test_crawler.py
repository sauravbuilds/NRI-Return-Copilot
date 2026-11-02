"""The Scrapy document fetcher: what it accepts, how it reports failures, and that it stays polite.
No network: responses and failures are built by hand."""
import asyncio

import pytest
import scrapy
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Response
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.python.failure import Failure

from scripts import crawler

URL = "https://example.org/doc.pdf"


@pytest.fixture
def spider():
    return crawler.OfficialDocumentSpider(targets={"doc": URL}, results={})


def reply(body, content_type="application/pdf", status=200):
    return Response(URL, body=body, status=status, headers={"Content-Type": content_type})


def failure_for(error):
    failure = Failure(error)
    failure.request = scrapy.Request(URL, cb_kwargs={"doc_id": "doc"})
    return failure


def test_a_pdf_response_is_stored_as_bytes(spider):
    spider.parse(reply(b"%PDF-1.7 content"), doc_id="doc")
    assert spider.results["doc"] == b"%PDF-1.7 content"


def test_an_html_page_is_a_failure_not_a_document(spider):
    spider.parse(reply(b"<html>Access Denied</html>", "text/html"), doc_id="doc")
    error = spider.results["doc"]
    assert isinstance(error, ValueError) and "not a PDF" in str(error) and "text/html" in str(error)


def test_http_errors_report_their_status(spider):
    spider.failed(failure_for(HttpError(reply(b"", status=403, content_type="text/html"))))
    assert str(spider.results["doc"]) == "HTTP 403"


def test_a_robots_exclusion_is_reported_and_not_retried_around(spider):
    spider.failed(failure_for(IgnoreRequest("Forbidden by robots.txt")))
    assert str(spider.results["doc"]) == "excluded by robots.txt"


def test_every_target_becomes_one_request(spider):
    async def collect():
        return [request async for request in spider.start()]
    requests = asyncio.run(collect())
    assert [r.url for r in requests] == [URL] and requests[0].cb_kwargs == {"doc_id": "doc"}


def test_the_crawler_is_polite_and_honest_about_who_it_is():
    settings = crawler.CRAWL_SETTINGS
    assert settings["ROBOTSTXT_OBEY"] is True
    assert settings["CONCURRENT_REQUESTS_PER_DOMAIN"] == 1 and settings["DOWNLOAD_DELAY"] >= 1
    assert settings["AUTOTHROTTLE_ENABLED"] is True
    assert "Mozilla" not in settings["USER_AGENT"] and "NRIReturnCopilot" in settings["USER_AGENT"]
