"""
Unit tests for ocr_app.py core logic (non-GUI components).
Run with: python3 -m pytest test_ocr_app.py -v
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from ocr_app import (
    TesseractEngine,
    EasyOCREngine,
    merge_results,
    preprocess_image,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image(text: str = "Hello World 123", size=(400, 80)):
    """Create a simple white image with black text for OCR testing."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), text, fill="black")
    return img


# ---------------------------------------------------------------------------
# TesseractEngine
# ---------------------------------------------------------------------------

class TestTesseractEngine:
    def test_detect_returns_list(self):
        engine = TesseractEngine()
        img = make_test_image()
        results = engine.detect(img)
        assert isinstance(results, list)

    def test_detect_finds_text(self):
        engine = TesseractEngine()
        img = make_test_image("TestWord")
        results = engine.detect(img)
        # Results must be a list; each item has the required schema
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r["text"], str) and r["text"]

    def test_result_schema(self):
        engine = TesseractEngine()
        img = make_test_image("ABC 123")
        results = engine.detect(img)
        for r in results:
            assert "text" in r
            assert "x" in r and "y" in r
            assert "w" in r and "h" in r
            assert "conf" in r
            assert 0.0 <= r["conf"] <= 1.0
            assert isinstance(r["text"], str)

    def test_low_confidence_filtered(self):
        engine = TesseractEngine()
        # Blank white image – should produce few/no results
        img = Image.new("RGB", (200, 50), color="white")
        results = engine.detect(img)
        for r in results:
            assert r["conf"] > 0.30


# ---------------------------------------------------------------------------
# EasyOCREngine
# ---------------------------------------------------------------------------

class TestEasyOCREngine:
    def test_detect_returns_list(self):
        engine = EasyOCREngine()
        img = make_test_image()
        results = engine.detect(img)
        assert isinstance(results, list)

    def test_result_schema(self):
        engine = EasyOCREngine()
        img = make_test_image("OCR")
        results = engine.detect(img)
        for r in results:
            assert "text" in r
            assert "x" in r and "y" in r
            assert "w" in r and "h" in r
            assert "conf" in r
            assert 0.0 <= r["conf"] <= 1.0

    def test_low_confidence_filtered(self):
        engine = EasyOCREngine()
        results = engine.detect(Image.new("RGB", (200, 50), "white"))
        for r in results:
            assert r["conf"] > 0.30


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------

class TestMergeResults:
    def _box(self, text, x, y, w, h, conf):
        return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": conf}

    def test_empty(self):
        assert merge_results([]) == []
        assert merge_results([[]]) == []

    def test_no_overlap(self):
        a = [self._box("A", 0, 0, 50, 20, 0.9)]
        b = [self._box("B", 200, 0, 50, 20, 0.8)]
        merged = merge_results([a, b])
        assert len(merged) == 2

    def test_duplicate_suppressed(self):
        # Two almost-identical boxes
        a = [self._box("hello", 10, 10, 50, 20, 0.9)]
        b = [self._box("hello", 12, 11, 50, 20, 0.8)]
        merged = merge_results([a, b])
        assert len(merged) == 1
        assert merged[0]["conf"] == 0.9  # higher confidence kept

    def test_high_iou_suppressed(self):
        # Boxes with IoU > 0.5 should be merged
        a = [self._box("word", 0, 0, 100, 30, 0.95)]
        b = [self._box("word", 5, 5, 100, 30, 0.70)]
        merged = merge_results([a, b])
        assert len(merged) == 1

    def test_sorted_by_confidence(self):
        # Highest confidence box should always be first in kept list
        boxes = [
            [self._box("A", 0, 0, 40, 20, 0.5)],
            [self._box("B", 100, 0, 40, 20, 0.99)],
            [self._box("C", 200, 0, 40, 20, 0.75)],
        ]
        merged = merge_results(boxes)
        confs = [r["conf"] for r in merged]
        assert confs == sorted(confs, reverse=True)

    def test_single_list(self):
        boxes = [self._box("X", i * 60, 0, 50, 20, 0.8) for i in range(4)]
        merged = merge_results([boxes])
        assert len(merged) == 4


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_returns_image(self):
        img = make_test_image()
        result = preprocess_image(img)
        assert isinstance(result, Image.Image)

    def test_same_size(self):
        img = make_test_image(size=(300, 100))
        result = preprocess_image(img)
        assert result.size == img.size

    def test_rgb_mode(self):
        img = Image.new("L", (100, 50), 128)  # grayscale
        result = preprocess_image(img)
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

class TestSupportedExtensions:
    def test_common_formats(self):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_non_image(self):
        assert ".txt" not in SUPPORTED_EXTENSIONS
        assert ".pdf" not in SUPPORTED_EXTENSIONS
