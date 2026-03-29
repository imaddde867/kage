from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingestion.parser import parse_markdown
import time

class VaultHandler(FileSystemEventHandler):
    def __init__(self, on_change):
        self.on_change = on_change

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self.on_change(Path(event.src_path), "modified")

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self.on_change(Path(event.src_path), "created")

def watch_vault(vault_path: str, on_change):
    observer = Observer()
    observer.schedule(VaultHandler(on_change), vault_path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
