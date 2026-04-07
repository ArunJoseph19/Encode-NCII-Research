"""
ArcFace wrapper using pure onnxruntime + opencv (no insightface package required).

Downloads the buffalo_l ONNX model pack from the InsightFace GitHub release on
first use and caches the two required files at ~/.insightface/models/buffalo_l/:

  det_10g.onnx   — SCRFD-10GF face detector (produces bounding boxes + 5-point
                   landmarks used for affine alignment)
  w600k_r50.onnx — ArcFace R50 recognition model (WebFace600K, 512-dim embedding)

Dependencies: onnxruntime, opencv-python-headless, numpy, urllib (stdlib).
"""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUFFALO_L_ZIP_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
_DEFAULT_CACHE_DIR = Path.home() / ".insightface" / "models" / "buffalo_l"

# 5-point reference template for 112×112 ArcFace alignment.
# Landmark order: left eye, right eye, nose tip, left mouth, right mouth.
_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Model download / cache
# ---------------------------------------------------------------------------

def _ensure_models(cache_dir: Path = _DEFAULT_CACHE_DIR) -> tuple[Path, Path]:
    """Download buffalo_l ONNX models if not already cached.

    Downloads the full buffalo_l.zip (~310 MB) from the InsightFace GitHub
    release, extracts only the two required files, then deletes the zip.

    Returns:
        (det_path, rec_path) — absolute paths to det_10g.onnx and w600k_r50.onnx.
    """
    det_path = cache_dir / "det_10g.onnx"
    rec_path = cache_dir / "w600k_r50.onnx"

    if det_path.exists() and rec_path.exists():
        return det_path, rec_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = cache_dir / "_buffalo_l_download.zip"

    print(f"Downloading buffalo_l models (~310 MB) → {cache_dir}")
    print(f"  Source: {_BUFFALO_L_ZIP_URL}")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(_BUFFALO_L_ZIP_URL, tmp_zip, reporthook=_progress)
        print()

        with zipfile.ZipFile(tmp_zip) as zf:
            for entry in zf.namelist():
                basename = Path(entry).name
                if basename in ("det_10g.onnx", "w600k_r50.onnx"):
                    dest = cache_dir / basename
                    dest.write_bytes(zf.read(entry))
                    print(f"  Extracted {basename} "
                          f"({dest.stat().st_size // (1024 * 1024):.0f} MB)")
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()

    if not det_path.exists() or not rec_path.exists():
        raise RuntimeError(
            f"Could not extract buffalo_l models from zip.\n"
            f"Manually download {_BUFFALO_L_ZIP_URL} and extract "
            f"det_10g.onnx + w600k_r50.onnx to {cache_dir}"
        )

    return det_path, rec_path


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class FaceResult:
    """A single detected face with its ArcFace embedding.

    Attributes match the insightface FaceAnalysis output so existing code
    that accesses .bbox and .normed_embedding works without modification.
    """
    __slots__ = ("bbox", "kps", "normed_embedding", "det_score")

    def __init__(
        self,
        bbox: np.ndarray,        # [x1, y1, x2, y2] float32, original image coords
        kps: np.ndarray,         # [5, 2] float32, original image coords
        normed_embedding: np.ndarray,  # [512] float32, L2-normalised
        det_score: float,
    ):
        self.bbox = bbox
        self.kps = kps
        self.normed_embedding = normed_embedding
        self.det_score = det_score


# ---------------------------------------------------------------------------
# SCRFD face detector (det_10g.onnx)
# ---------------------------------------------------------------------------

class _SCRFDDetector:
    """SCRFD-10GF face detector.

    Decodes the multi-scale outputs of det_10g.onnx into bounding boxes and
    5-point facial landmarks in original image coordinates.

    Output layout of det_10g.onnx (9 outputs, FMC=3 scales):
      [0-2]  score_8, score_16, score_32   — sigmoid confidence, shape [N, 1]
      [3-5]  bbox_8,  bbox_16,  bbox_32    — ltrb distances in stride units, [N, 4]
      [6-8]  kps_8,   kps_16,   kps_32    — keypoint offsets in stride units, [N, 10]
    """

    _STRIDES = [8, 16, 32]
    _FMC = 3          # number of feature-map scales
    _NUM_ANCHORS = 2  # SCRFD uses 2 anchors per spatial location

    def __init__(
        self,
        model_path: Path,
        input_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.4,
    ):
        providers = _ort_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_w, self.input_h = input_size
        self.det_thresh = det_thresh

        num_outputs = len(self.session.get_outputs())
        self._use_kps = (num_outputs == self._FMC * 3)

    def _anchors_for_stride(self, stride: int) -> np.ndarray:
        """Return anchor centres [H*W*num_anchors, 2] for one stride level."""
        h = self.input_h // stride
        w = self.input_w // stride
        xs = np.arange(w, dtype=np.float32) * stride
        ys = np.arange(h, dtype=np.float32) * stride
        grid_x, grid_y = np.meshgrid(xs, ys)
        centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        return np.repeat(centers, self._NUM_ANCHORS, axis=0)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
        return keep

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Detect faces in a BGR image.

        Returns list of dicts with keys: bbox [4], score float, kps [5, 2].
        Coordinates are in original image pixel space.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        iw, ih = self.input_w, self.input_h
        scale_x = orig_w / iw
        scale_y = orig_h / ih

        # Preprocess: resize → normalise → NCHW
        resized = cv2.resize(img_bgr, (iw, ih))
        blob = (resized.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

        outputs = self.session.run(None, {self.input_name: blob})

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []

        for i, stride in enumerate(self._STRIDES):
            scores = outputs[i].flatten()                   # sigmoid scores in [0, 1]
            bbox_pred = outputs[i + self._FMC]              # [N, 4], stride units
            anchors = self._anchors_for_stride(stride)

            mask = scores >= self.det_thresh
            if not mask.any():
                continue

            scores = scores[mask]
            bbox_pred = bbox_pred[mask] * stride
            centers = anchors[mask]

            # Decode ltrb → xyxy in resized-image coords, then scale to original
            x1 = (centers[:, 0] - bbox_pred[:, 0]) * scale_x
            y1 = (centers[:, 1] - bbox_pred[:, 1]) * scale_y
            x2 = (centers[:, 0] + bbox_pred[:, 2]) * scale_x
            y2 = (centers[:, 1] + bbox_pred[:, 3]) * scale_y
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            all_boxes.append(boxes)
            all_scores.append(scores)

            if self._use_kps:
                kps_pred = outputs[i + self._FMC * 2][mask] * stride  # [N, 10]
                kps = np.zeros((len(scores), 5, 2), dtype=np.float32)
                for k in range(5):
                    kps[:, k, 0] = (centers[:, 0] + kps_pred[:, k * 2])     * scale_x
                    kps[:, k, 1] = (centers[:, 1] + kps_pred[:, k * 2 + 1]) * scale_y
                all_kps.append(kps)

        if not all_boxes:
            return []

        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_kps = np.concatenate(all_kps) if all_kps else np.zeros((len(all_boxes), 5, 2))

        keep = self._nms(all_boxes, all_scores)
        return [
            {"bbox": all_boxes[k], "score": float(all_scores[k]), "kps": all_kps[k]}
            for k in keep
        ]


# ---------------------------------------------------------------------------
# ArcFace recognizer (w600k_r50.onnx)
# ---------------------------------------------------------------------------

class _ArcFaceRecognizer:
    """ArcFace R50 embedding model.

    Aligns a detected face to 112×112 using an affine transform derived from
    the 5 facial keypoints, then runs the ONNX model to produce a 512-dim
    L2-normalised embedding.
    """

    def __init__(self, model_path: Path):
        providers = _ort_providers()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _align(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Affine-warp face region to 112×112 using 5-point landmarks."""
        M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if M is None:
            # Fallback: centre-crop and resize when landmark fit fails
            return cv2.resize(img_bgr, (112, 112))
        return cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)

    def embed(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Return a 512-dim L2-normalised ArcFace embedding for one face.

        Args:
            img_bgr: Full BGR image (uint8).
            kps: 5-point landmarks [5, 2] in image pixel coords.

        Returns:
            [512] float32 embedding, L2-normalised.
        """
        aligned = self._align(img_bgr, kps)
        # ArcFace preprocessing: BGR → RGB, scale to [-1, 1], NCHW
        face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        blob = (face_rgb.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 112, 112]
        embedding = self.session.run(None, {self.input_name: blob})[0][0]  # [512]
        norm = np.linalg.norm(embedding)
        return (embedding / (norm + 1e-6)).astype(np.float32)


# ---------------------------------------------------------------------------
# Provider helper
# ---------------------------------------------------------------------------

def _ort_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available]


# ---------------------------------------------------------------------------
# Combined analyser — drop-in for insightface FaceAnalysis
# ---------------------------------------------------------------------------

class FaceAnalyzerONNX:
    """Combined face detection + ArcFace recognition, no insightface required.

    Public interface:
        face_app = FaceAnalyzerONNX()
        results  = face_app.get(img_bgr)   # list[FaceResult], largest face first

    Each FaceResult exposes .bbox ([x1,y1,x2,y2]), .kps ([5,2]),
    .normed_embedding ([512]), and .det_score — matching the subset of the
    insightface API used in this codebase.

    Models are downloaded automatically from the InsightFace CDN on first use
    and cached at ~/.insightface/models/buffalo_l/.
    """

    def __init__(
        self,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        input_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.4,
    ):
        det_path, rec_path = _ensure_models(cache_dir)
        self._detector = _SCRFDDetector(det_path, input_size=input_size,
                                        det_thresh=det_thresh)
        self._recognizer = _ArcFaceRecognizer(rec_path)

    def get(self, img_bgr: np.ndarray) -> list[FaceResult]:
        """Detect all faces and return their ArcFace embeddings.

        Args:
            img_bgr: OpenCV-format uint8 BGR image.

        Returns:
            List of FaceResult sorted by face area (largest first).
            Empty list if no face detected.
        """
        detections = self._detector.detect(img_bgr)
        results = []
        for det in detections:
            emb = self._recognizer.embed(img_bgr, det["kps"])
            results.append(FaceResult(
                bbox=det["bbox"],
                kps=det["kps"],
                normed_embedding=emb,
                det_score=det["score"],
            ))
        # Sort largest face first to match insightface behaviour
        results.sort(
            key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
            reverse=True,
        )
        return results


# ---------------------------------------------------------------------------
# Public ArcFaceWrapper (same batch_similarity interface as before)
# ---------------------------------------------------------------------------

class ArcFaceWrapper:
    """Wrapper around FaceAnalyzerONNX for identity similarity evaluation.

    Provides the same batch_similarity(generated_dir, reference_dir) interface
    as the previous insightface-based version so 04_arcface_similarity.py
    requires no changes.
    """

    def __init__(
        self,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        input_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.4,
    ):
        self._analyzer = FaceAnalyzerONNX(cache_dir=cache_dir,
                                          input_size=input_size,
                                          det_thresh=det_thresh)

    def get_embedding(self, image) -> Optional[np.ndarray]:
        """Extract 512-d face embedding from a PIL Image or BGR ndarray.

        Returns:
            [512] normalised embedding, or None if no face detected.
        """
        if not isinstance(image, np.ndarray):
            # PIL Image → BGR ndarray
            img_array = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image

        faces = self._analyzer.get(img_bgr)
        if not faces:
            return None
        return faces[0].normed_embedding  # largest face

    def get_embeddings_from_dir(
        self,
        image_dir,
        max_images: int = 100,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Extract embeddings for all faces in a directory.

        Returns:
            (embeddings, filenames) — images with no detected face are skipped.
        """
        from PIL import Image as _PILImage

        image_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in extensions
        )[:max_images]

        embeddings, filenames = [], []
        for path in tqdm(image_paths, desc="Extracting face embeddings"):
            img = _PILImage.open(path).convert("RGB")
            emb = self.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                filenames.append(path.name)
            else:
                print(f"  Warning: No face detected in {path.name}")

        return embeddings, filenames

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return float(dot / norm) if norm else 0.0

    def batch_similarity(
        self,
        generated_dir,
        reference_dir,
    ) -> dict:
        """Compute identity similarity between generated and reference images.

        For each generated image, computes cosine similarity against all
        reference embeddings and reports max & mean similarity.

        Returns:
            Dict with per-image scores, aggregates, and pass/fail verdict.
        """
        print("Processing generated images...")
        gen_embeddings, gen_files = self.get_embeddings_from_dir(generated_dir)
        print(f"  Got {len(gen_embeddings)} embeddings from {generated_dir}")

        print("Processing reference images...")
        ref_embeddings, ref_files = self.get_embeddings_from_dir(reference_dir)
        print(f"  Got {len(ref_embeddings)} embeddings from {reference_dir}")

        if not gen_embeddings or not ref_embeddings:
            return {
                "error": "Could not extract embeddings from one or both directories",
                "generated_faces_found": len(gen_embeddings),
                "reference_faces_found": len(ref_embeddings),
            }

        ref_matrix = np.stack(ref_embeddings)  # [N_ref, 512]

        per_image_results = []
        for gen_emb, gen_file in zip(gen_embeddings, gen_files):
            similarities = ref_matrix @ gen_emb  # [N_ref] — embeddings are normalised
            per_image_results.append({
                "filename": gen_file,
                "max_similarity": float(np.max(similarities)),
                "mean_similarity": float(np.mean(similarities)),
                "best_match_ref": ref_files[int(np.argmax(similarities))],
            })

        max_sims = [r["max_similarity"] for r in per_image_results]
        mean_sims = [r["mean_similarity"] for r in per_image_results]

        return {
            "model": "buffalo_l/w600k_r50.onnx (ArcFace R50)",
            "num_generated": len(gen_embeddings),
            "num_reference": len(ref_embeddings),
            "per_image": per_image_results,
            "aggregate": {
                "mean_max_similarity": float(np.mean(max_sims)),
                "std_max_similarity": float(np.std(max_sims)),
                "mean_mean_similarity": float(np.mean(mean_sims)),
                "median_max_similarity": float(np.median(max_sims)),
                "min_max_similarity": float(np.min(max_sims)),
                "max_max_similarity": float(np.max(max_sims)),
            },
            "threshold": 0.45,
            "identity_preserved": bool(np.mean(max_sims) > 0.45),
        }
