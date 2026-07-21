"""Print a compact, content-aware inventory of the official SHREC'19 ZIP."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import PurePosixPath
from zipfile import ZipFile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        directories = sorted({str(PurePosixPath(item.filename).parent) for item in files})
        extensions = Counter(PurePosixPath(item.filename).suffix.lower() for item in files)
        non_obj = sorted(item.filename for item in files if not item.filename.lower().endswith(".obj"))
        text_content = {}
        for name in non_obj:
            if PurePosixPath(name).suffix.lower() in {".txt", ".csv", ".json", ".md"}:
                raw = archive.read(name)
                text_content[name] = raw.decode("utf-8", errors="replace")
        payload = {
            "file_count": len(files),
            "total_uncompressed_bytes": sum(item.file_size for item in files),
            "extensions": dict(sorted(extensions.items())),
            "directories": directories,
            "non_obj_files": non_obj,
            "text_content": text_content,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
