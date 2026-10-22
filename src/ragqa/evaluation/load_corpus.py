"""Upload the prepared corpus through the API and write the ingest manifest.

    python -m ragqa.evaluation.load_corpus --api http://localhost:8000 \\
        --corpus ~/data/corpora/kubernetes/corpus --index base \\
        --releases release-1.32 release-1.34 release-1.36 --out eval/corpus/ingest-base.json

Releases are loaded in order, so each page's registry version numbers follow the release order.
A page whose content is identical to the previous release is not a new version (story B1); the
manifest then points that release at the existing version. The API runs the ingestion in the
background, so each release is awaited until every job is `ready` before the next one starts.
This is a long job (every page is chunked and embedded): start it on purpose.
"""

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from ragqa.evaluation.manifest import IngestManifest, ManifestVersion
from ragqa.ingestion.parsers import default_registry


def pages(corpus: Path, release: str) -> list[str]:
    """Page paths below the release folder, sorted."""
    root = corpus / release
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix in {".md", ".docx"})


class Loader:
    def __init__(
        self,
        http: httpx.AsyncClient,
        corpus: Path,
        index: str,
        authors: dict[str, str],
        *,
        concurrency: int = 4,
        poll_s: float = 2.0,
    ) -> None:
        self._http = http
        self._corpus = corpus
        self._index = index
        self._authors = authors
        self._gate = asyncio.Semaphore(concurrency)
        self._poll_s = poll_s
        self._parsers = default_registry()
        self.documents: dict[str, uuid.UUID] = {}
        self.latest: dict[str, ManifestVersion] = {}
        self.versions: list[ManifestVersion] = []

    async def _upload(self, release: str, page: str) -> tuple[str, dict[str, Any]]:
        path = self._corpus / release / page
        data = path.read_bytes()
        metadata = {"author": self._authors[page]} if page in self._authors else {}
        files = {"file": (path.name, data)}
        form = {"metadata": json.dumps(metadata), "index": self._index}
        async with self._gate:
            if page in self.documents:
                reply = await self._http.post(
                    f"/documents/{self.documents[page]}/versions", files=files, data=form
                )
            else:
                title = self._parsers.for_filename(path.name).parse(data, filename=path.name).title
                reply = await self._http.post(
                    "/documents", files=files, data={**form, "title": title}
                )
        if reply.status_code not in (200, 202):
            raise RuntimeError(f"{release}/{page}: http {reply.status_code}: {reply.text[:300]}")
        return page, reply.json()

    async def load_release(self, release: str) -> int:
        results = await asyncio.gather(
            *(self._upload(release, p) for p in pages(self._corpus, release))
        )
        waiting: dict[uuid.UUID, str] = {}
        for page, body in results:
            if body["status"] == "no_changes":
                previous = self.latest[page]
                self.versions.append(previous.model_copy(update={"release": release}))
                continue
            document_id, version_id = uuid.UUID(body["document_id"]), uuid.UUID(body["version_id"])
            self.documents[page] = document_id
            entry = ManifestVersion(
                document=page,
                release=release,
                document_id=document_id,
                version_id=version_id,
                version=body["version"],
            )
            self.latest[page] = entry
            self.versions.append(entry)
            waiting[uuid.UUID(body["job_id"])] = page
        await self._wait(waiting)
        return len(results)

    async def _wait(self, jobs: dict[uuid.UUID, str]) -> None:
        pending = dict(jobs)
        while pending:
            await asyncio.sleep(self._poll_s)
            for job_id in list(pending):
                body = (await self._http.get(f"/jobs/{job_id}")).json()
                if body["status"] == "ready":
                    del pending[job_id]
                elif body["status"] in ("failed", "dead"):
                    raise RuntimeError(f"{pending[job_id]}: job {body['status']}: {body['error']}")
            print(f"  {len(jobs) - len(pending)}/{len(jobs)} jobs ready", file=sys.stderr)

    def manifest(self, releases: Sequence[str]) -> IngestManifest:
        return IngestManifest(
            index=self._index, corpus_releases=tuple(releases), versions=tuple(self.versions)
        )


async def amain(args: argparse.Namespace) -> int:
    authors: dict[str, str] = {}
    if args.authors and args.authors.exists():
        authors = json.loads(args.authors.read_text(encoding="utf-8"))["mapping"]
    if args.out.exists():
        print(f"{args.out} exists; choose another file", file=sys.stderr)
        return 1
    async with httpx.AsyncClient(base_url=args.api, timeout=120.0) as http:
        loader = Loader(http, args.corpus, args.index, authors, concurrency=args.concurrency)
        for release in args.releases:
            count = await loader.load_release(release)
            print(f"{release}: {count} pages loaded", file=sys.stderr)
        loader.manifest(args.releases).save(args.out)
    print(f"manifest written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload the corpus and write the ingest manifest.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--releases", nargs="+", required=True)
    parser.add_argument("--authors", type=Path, default=Path("eval/corpus/authors.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(amain(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
