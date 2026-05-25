import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Iterable, List, Optional, Tuple

LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str


def parse_log_line(line: str) -> Optional[LogEntry]:
    # Expected format: "YYYY-MM-DD HH:MM:SS,mmm [LEVEL] message"
    line = line.strip()
    if not line:
        return None
    try:
        time_part, rest = line.split(" ", 1)
        date_part, rest = rest.split(" ", 1)
        timestamp = datetime.strptime(f"{time_part} {date_part}", LOG_TIME_FORMAT)
        if not rest.startswith("["):
            return None
        level_end = rest.find("]")
        if level_end == -1:
            return None
        level = rest[1:level_end].strip()
        message = rest[level_end + 1 :].strip()
        return LogEntry(timestamp=timestamp, level=level, message=message)
    except Exception:
        return None


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def filter_entries(
    entries: Iterable[LogEntry],
    levels: Optional[List[str]],
    search: Optional[str],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> List[LogEntry]:
    filtered: List[LogEntry] = []
    search_lower = search.lower() if search else None
    levels_set = {lvl.upper() for lvl in levels} if levels else None

    for entry in entries:
        if levels_set and entry.level.upper() not in levels_set:
            continue
        if start_time and entry.timestamp < start_time:
            continue
        if end_time and entry.timestamp > end_time:
            continue
        if search_lower and search_lower not in entry.message.lower():
            continue
        filtered.append(entry)

    return filtered


def read_new_entries(path: Path, offset: int) -> Tuple[List[LogEntry], int]:
    if not path.exists():
        return [], offset

    entries: List[LogEntry] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        for line in handle:
            entry = parse_log_line(line)
            if entry:
                entries.append(entry)
        new_offset = handle.tell()

    return entries, new_offset


def render(entries: List[LogEntry], log_path: Path, follow: bool) -> None:
    term_width = os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 120
    header = f"Log UI | {log_path} | entries: {len(entries)} | follow: {follow}"
    print(header[:term_width])
    print("-" * min(term_width, len(header)))
    for entry in entries:
        line = f"{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} [{entry.level}] {entry.message}"
        if len(line) <= term_width:
            print(line)
        else:
            print(line[: term_width - 3] + "...")


def export_entries(entries: List[LogEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "level": e.level,
                        "message": e.message,
                    }
                    for e in entries
                ],
                handle,
                indent=2,
            )
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "level", "message"])
        for entry in entries:
            writer.writerow([entry.timestamp.isoformat(), entry.level, entry.message])


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal log UI for Project Kuru.")
    parser.add_argument("--log", default="kuru.log", help="Path to the log file.")
    parser.add_argument("--levels", default="", help="Comma-separated list of levels (INFO,WARN,ERROR).")
    parser.add_argument("--search", default="", help="Case-insensitive search string.")
    parser.add_argument("--start", default="", help="Start time (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]).")
    parser.add_argument("--end", default="", help="End time (YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]).")
    parser.add_argument("--follow", action="store_true", help="Tail the log file.")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--max-entries", type=int, default=5000, help="Max entries to retain in memory.")
    parser.add_argument("--export", default="", help="Export filtered logs to CSV/JSON path and exit.")
    args = parser.parse_args()

    log_path = Path(args.log)
    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    search = args.search.strip() or None
    start_time = parse_time(args.start)
    end_time = parse_time(args.end)

    offset = 0
    buffer: Deque[LogEntry] = deque(maxlen=args.max_entries)

    while True:
        new_entries, offset = read_new_entries(log_path, offset)
        buffer.extend(new_entries)

        filtered = filter_entries(buffer, levels, search, start_time, end_time)

        if args.export:
            export_entries(filtered, Path(args.export))
            return 0

        os.system("cls" if os.name == "nt" else "clear")
        render(filtered, log_path, args.follow)

        if not args.follow:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
