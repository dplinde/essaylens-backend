"""Image pre-processing for handwritten essay photos: brightness/contrast
normalisation, denoising, adaptive binarisation, deskew, and text-line segmentation."""
import cv2
import numpy as np


def load_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def enhance(img: np.ndarray) -> np.ndarray:
    """Return a clean binary (white background, black ink) image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) > 2200:  # cap size for speed
        s = 2200 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # contrast
    gray = clahe.apply(gray)
    bg = cv2.medianBlur(gray, 51)  # remove uneven lighting
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.fastNlMeansDenoising(norm, None, 10, 7, 21)
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return deskew(binary)


def deskew(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.where(binary < 128)
    if len(xs) < 200:
        return binary
    coords = np.column_stack((xs, ys)).astype(np.float32)
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return binary
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def segment_lines(binary: np.ndarray, min_height: int = 12) -> list[np.ndarray]:
    """Split a page into text-line crops using horizontal projection."""
    ink = (binary < 128).astype(np.uint8)
    proj = ink.sum(axis=1)
    thresh = max(2, int(0.01 * ink.shape[1]))
    lines, start = [], None
    for y, v in enumerate(proj):
        if v > thresh and start is None:
            start = y
        elif v <= thresh and start is not None:
            if y - start >= min_height:
                lines.append(binary[max(0, start - 4): y + 4])
            start = None
    if start is not None and len(proj) - start >= min_height:
        lines.append(binary[start:])
    return lines


def cursive_score(binary: np.ndarray) -> float:
    """Heuristic 0..1: how connected/cursive the writing looks.
    Block print has many separate connected components per line; cursive has few, long ones."""
    ink = (binary < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 10:
        return 1.0
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = area > 15
    if keep.sum() < 10:
        return 1.0
    ratio = np.median(widths[keep] / np.maximum(heights[keep], 1))
    long_frac = float(np.mean(widths[keep] > 3.0 * np.median(heights[keep])))
    return float(min(1.0, max(0.0, 0.5 * (ratio - 1.0) / 2.0 + 0.5 * long_frac * 3)))
