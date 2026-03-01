"""
ArcFace wrapper for identity similarity evaluation.

Uses InsightFace's buffalo_l model to compute 512-dimensional face
embeddings and cosine similarity between generated and reference images.
This is the primary identity preservation metric for POC 1.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


class ArcFaceWrapper:
    """Wrapper around InsightFace ArcFace for face embedding and comparison.

    Attributes:
        model: InsightFace FaceAnalysis model.
        det_size: Detection input size.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (320, 320),
        det_thresh: float = 0.3,
        providers: Optional[list[str]] = None,
    ):
        """Initialise ArcFace face analysis model.

        Args:
            model_name: InsightFace model pack name.
            det_size: Face detection input resolution (lower = more sensitive to small faces).
            det_thresh: Face detection confidence threshold (lower = more detections).
            providers: ONNX Runtime execution providers.
        """
        import insightface
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        self.det_size = det_size

    def get_embedding(self, image: Image.Image | np.ndarray) -> Optional[np.ndarray]:
        """Extract 512-d face embedding from an image.

        Args:
            image: PIL Image or numpy array (BGR).

        Returns:
            512-dimensional normalised embedding, or None if no face detected.
        """
        import cv2

        if isinstance(image, Image.Image):
            # Convert PIL RGB to OpenCV BGR
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image

        faces = self.app.get(img_bgr)

        if not faces:
            return None

        # Use the largest detected face
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = largest_face.normed_embedding

        return embedding

    def get_embeddings_from_dir(
        self,
        image_dir: str | Path,
        max_images: int = 100,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Extract embeddings for all faces in a directory.

        Args:
            image_dir: Path to directory containing face images.
            max_images: Maximum images to process.

        Returns:
            Tuple of (embeddings_list, filenames_list).
            Images where no face is detected are skipped.
        """
        image_dir = Path(image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in extensions
        )[:max_images]

        embeddings = []
        filenames = []

        for path in tqdm(image_paths, desc="Extracting face embeddings"):
            img = Image.open(path).convert("RGB")
            emb = self.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
                filenames.append(path.name)
            else:
                print(f"  Warning: No face detected in {path.name}")

        return embeddings, filenames

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def batch_similarity(
        self,
        generated_dir: str | Path,
        reference_dir: str | Path,
    ) -> dict:
        """Compute identity similarity between generated and reference images.

        For each generated image, computes cosine similarity against all
        reference images and reports max & mean similarity.

        Args:
            generated_dir: Directory of generated images.
            reference_dir: Directory of reference (original) images.

        Returns:
            Dict with per-image scores, aggregates, and pass/fail.
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

        # Build reference embedding matrix
        ref_matrix = np.stack(ref_embeddings)  # (N_ref, 512)

        per_image_results = []
        for gen_emb, gen_file in zip(gen_embeddings, gen_files):
            # Cosine similarity against all references
            similarities = ref_matrix @ gen_emb  # (N_ref,) — already normalised
            per_image_results.append({
                "filename": gen_file,
                "max_similarity": float(np.max(similarities)),
                "mean_similarity": float(np.mean(similarities)),
                "best_match_ref": ref_files[int(np.argmax(similarities))],
            })

        max_sims = [r["max_similarity"] for r in per_image_results]
        mean_sims = [r["mean_similarity"] for r in per_image_results]

        return {
            "model": "insightface/buffalo_l",
            "num_generated": len(gen_embeddings),
            "num_generated_no_face": len(gen_files) - len(gen_embeddings) if hasattr(gen_files, '__len__') else 0,
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
