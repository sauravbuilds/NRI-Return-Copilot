"""Polite document fetcher built on Scrapy.

Fetches the PDFs listed in the manifest. The crawler identifies itself honestly, obeys
robots.txt, keeps to one request at a time per site with a delay, and retries transient
errors. A response is accepted only if it is a PDF. Anything else (an HTML page from a
site's bot protection, a moved link, a robots.txt exclusion) is reported as a failure for
that document so the existing file is kept and the document can be added manually.
"""
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import IgnoreRequest
from scrapy.spidermiddlewares.httperror import HttpError

import config

USER_AGENT = "NRIReturnCopilot-refresh/1.0"
MAX_PDF_BYTES = 100 * 1024 * 1024

CRAWL_SETTINGS = {
    "USER_AGENT": USER_AGENT,
    "ROBOTSTXT_OBEY": True,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
    "DOWNLOAD_DELAY": 2,
    "AUTOTHROTTLE_ENABLED": True,
    "RETRY_TIMES": 2,
    "DOWNLOAD_TIMEOUT": config.DOWNLOAD_TIMEOUT_SECONDS,
    "DOWNLOAD_MAXSIZE": MAX_PDF_BYTES,
    "COOKIES_ENABLED": False,
    "TELNETCONSOLE_ENABLED": False,
    "LOG_LEVEL": "WARNING",
}


class OfficialDocumentSpider(scrapy.Spider):
    """Fetches each (doc_id, url) once and stores the PDF bytes, or the reason it failed,
    in `results[doc_id]`."""
    name = "official_documents"

    def __init__(self, targets: dict[str, str], results: dict, **kwargs):
        super().__init__(**kwargs)
        self.targets, self.results = targets, results

    async def start(self):
        for doc_id, url in self.targets.items():
            yield scrapy.Request(url, callback=self.parse, errback=self.failed,
                                 cb_kwargs={"doc_id": doc_id}, dont_filter=True)

    def parse(self, response, doc_id: str):
        if response.body.startswith(b"%PDF"):
            self.results[doc_id] = response.body
        else:
            kind = response.headers.get("Content-Type", b"").decode("latin-1") or "unknown type"
            self.results[doc_id] = ValueError(f"response is not a PDF ({kind}); blocked or moved?")

    def failed(self, failure):
        doc_id = failure.request.cb_kwargs["doc_id"]
        error = failure.value
        if isinstance(error, HttpError):
            reason = f"HTTP {error.response.status}"
        elif isinstance(error, IgnoreRequest):
            reason = "excluded by robots.txt"
        else:
            reason = type(error).__name__
        self.results[doc_id] = RuntimeError(reason)


def fetch_pdfs(targets: dict[str, str]) -> dict:
    """Fetch every url in `targets` ({doc_id: url}). Returns {doc_id: bytes | Exception}."""
    results: dict = {}
    process = CrawlerProcess(CRAWL_SETTINGS)
    process.crawl(OfficialDocumentSpider, targets=targets, results=results)
    process.start()
    return {doc_id: results.get(doc_id, RuntimeError("no response")) for doc_id in targets}
