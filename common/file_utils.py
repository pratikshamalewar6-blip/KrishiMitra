"""
KrishiMitra - File Utilities

Production-ready utility functions for file and directory operations.

Features
--------
- Directory management
- JSON read/write
- CSV export
- Image discovery
- File copy/move/delete
- File size calculation
- Recursive image search

Author:
Pratiksha Malewar
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from common.logger import LoggerManager


class FileUtils:
    """
    Utility class for file operations.

    All methods are static because no object state is required.
    """

    logger = LoggerManager.get_logger("FileUtils")

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    # ==========================================================
    # Directory Operations
    # ==========================================================

    @staticmethod
    def ensure_directory(path: str | Path) -> Path:
        """
        Create directory if it does not exist.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Path
        """

        directory = Path(path)

        directory.mkdir(parents=True, exist_ok=True)

        FileUtils.logger.info(f"Directory ready: {directory}")

        return directory

    @staticmethod
    def directory_exists(path: str | Path) -> bool:
        """
        Check whether directory exists.
        """

        return Path(path).exists()

    @staticmethod
    def delete_directory(path: str | Path) -> None:
        """
        Delete directory recursively.
        """

        directory = Path(path)

        if directory.exists():

            shutil.rmtree(directory)

            FileUtils.logger.info(f"Deleted directory: {directory}")

    # ==========================================================
    # File Operations
    # ==========================================================

    @staticmethod
    def copy_file(source: str | Path, destination: str | Path) -> None:
        """
        Copy file.
        """

        src = Path(source)
        dst = Path(destination)

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)

        FileUtils.logger.info(f"Copied {src} -> {dst}")

    @staticmethod
    def move_file(source: str | Path, destination: str | Path) -> None:
        """
        Move file.
        """

        src = Path(source)
        dst = Path(destination)

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))

        FileUtils.logger.info(f"Moved {src} -> {dst}")

    @staticmethod
    def delete_file(path: str | Path) -> None:
        """
        Delete file.
        """

        file_path = Path(path)

        if file_path.exists():

            file_path.unlink()

            FileUtils.logger.info(f"Deleted file: {file_path}")

    @staticmethod
    def rename_file(source: str | Path, destination: str | Path) -> None:
        """
        Rename file.
        """

        src = Path(source)
        dst = Path(destination)

        src.rename(dst)

        FileUtils.logger.info(f"Renamed {src} -> {dst}")

    # ==========================================================
    # JSON
    # ==========================================================

    @staticmethod
    def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
        """
        Save dictionary as JSON.
        """

        output = Path(output_path)

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        FileUtils.logger.info(f"JSON saved: {output}")

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        """
        Load JSON.
        """

        with open(path, "r", encoding="utf-8") as file:

            data = json.load(file)

        FileUtils.logger.info(f"JSON loaded: {path}")

        return data

    # ==========================================================
    # CSV
    # ==========================================================

    @staticmethod
    def save_csv(
        rows: List[List[Any]],
        headers: List[str],
        output_path: str | Path,
    ) -> None:
        """
        Save CSV file.
        """

        output = Path(output_path)

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="", encoding="utf-8") as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(headers)

            writer.writerows(rows)

        FileUtils.logger.info(f"CSV saved: {output}")

    # ==========================================================
    # Image Discovery
    # ==========================================================

    @staticmethod
    def list_images(
        directory: str | Path,
        recursive: bool = True,
    ) -> List[Path]:
        """
        Find all images inside a directory.

        Parameters
        ----------
        directory : str | Path

        recursive : bool

        Returns
        -------
        List[Path]
        """

        directory = Path(directory)

        if not directory.exists():

            FileUtils.logger.warning(f"Directory not found: {directory}")

            return []

        if recursive:

            files = directory.rglob("*")

        else:

            files = directory.glob("*")

        images = [
            file
            for file in files
            if file.suffix.lower() in FileUtils.IMAGE_EXTENSIONS
        ]

        FileUtils.logger.info(f"Found {len(images)} images in {directory}")

        return sorted(images)

    # ==========================================================
    # File Information
    # ==========================================================

    @staticmethod
    def get_file_size(path: str | Path) -> float:
        """
        File size in MB.
        """

        file_path = Path(path)

        size = file_path.stat().st_size

        return round(size / (1024 * 1024), 4)

    @staticmethod
    def file_exists(path: str | Path) -> bool:
        """
        Check whether file exists.
        """

        return Path(path).exists()

    @staticmethod
    def count_files(
        directory: str | Path,
        recursive: bool = True,
    ) -> int:
        """
        Count all files inside directory.
        """

        directory = Path(directory)

        if not directory.exists():

            return 0

        if recursive:

            return sum(1 for _ in directory.rglob("*") if _.is_file())

        return sum(1 for _ in directory.glob("*") if _.is_file())