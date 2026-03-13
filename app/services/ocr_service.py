from __future__ import annotations

import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.image_preprocess import preprocess_image_bytes
from app.services.layout_segmenter import segment_formula_regions


@dataclass
class OCRResult:
    text: str
    engine: str
    segments: list[dict[str, Any]]
    saved_path: str | None = None
    error: str | None = None
    preprocessed: bool = False


class OCRService:
    def __init__(self) -> None:
        self._p2t = None
        self._p2t_unavailable = False
        self._cnocr = None
        self._latex_ocr = None
        self._dual_unavailable = False

        self._rapid_model = None
        self._pix2tex_model = None
        self._rapid_unavailable = False
        self._pix2tex_unavailable = False

        base_dir = Path(__file__).resolve().parents[2]
        self._output_dir = base_dir / "outputs" / "ocr_runs"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = base_dir / ".cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_error: str | None = None

    def extract(self, image_bytes: bytes, fallback_text: str | None = None) -> OCRResult:
        self.last_error = None
        if fallback_text and fallback_text.strip():
            text = fallback_text.strip().replace("\\r\\n", "\n").replace("\\n", "\n")
            segments = self._segments_from_text(text)
            saved_path = self._save_ocr_run(
                engine="manual_text",
                raw_text=text,
                segments=segments,
            )
            return OCRResult(
                text=text,
                engine="manual_text",
                segments=segments,
                saved_path=saved_path,
                error=None,
                preprocessed=False,
            )

        processed_bytes = preprocess_image_bytes(image_bytes)
        preprocessed = processed_bytes != image_bytes

        dual_text, dual_segments = self._extract_with_dual_pipeline(processed_bytes)
        if dual_text:
            saved_path = self._save_ocr_run(
                engine="dual-cnocr-latexocr",
                raw_text=dual_text,
                segments=dual_segments,
            )
            return OCRResult(
                text=dual_text,
                engine="dual-cnocr-latexocr",
                segments=dual_segments,
                saved_path=saved_path,
                error=None,
                preprocessed=preprocessed,
            )

        p2t_text, p2t_segments = self._extract_with_pix2text(processed_bytes)
        if p2t_text:
            saved_path = self._save_ocr_run(
                engine="pix2text",
                raw_text=p2t_text,
                segments=p2t_segments,
            )
            return OCRResult(
                text=p2t_text,
                engine="pix2text",
                segments=p2t_segments,
                saved_path=saved_path,
                error=None,
                preprocessed=preprocessed,
            )

        # Fallbacks keep old path for minimal availability.
        segmented_text = self._extract_with_layout_pipeline(processed_bytes)
        if segmented_text:
            segments = self._segments_from_text(segmented_text)
            saved_path = self._save_ocr_run(
                engine="rapid-latex-ocr+layout",
                raw_text=segmented_text,
                segments=segments,
            )
            return OCRResult(
                text=segmented_text,
                engine="rapid-latex-ocr+layout",
                segments=segments,
                saved_path=saved_path,
                error=None,
                preprocessed=preprocessed,
            )

        rapid_latex = self._extract_with_rapid_latex_ocr(processed_bytes)
        if rapid_latex:
            segments = self._segments_from_text(rapid_latex)
            saved_path = self._save_ocr_run(
                engine="rapid-latex-ocr",
                raw_text=rapid_latex,
                segments=segments,
            )
            return OCRResult(
                text=rapid_latex,
                engine="rapid-latex-ocr",
                segments=segments,
                saved_path=saved_path,
                error=None,
                preprocessed=preprocessed,
            )

        pix2tex = self._extract_with_pix2tex(processed_bytes)
        if pix2tex:
            segments = self._segments_from_text(pix2tex)
            saved_path = self._save_ocr_run(
                engine="pix2tex",
                raw_text=pix2tex,
                segments=segments,
            )
            return OCRResult(
                text=pix2tex,
                engine="pix2tex",
                segments=segments,
                saved_path=saved_path,
                error=None,
                preprocessed=preprocessed,
            )

        return OCRResult(
            text="",
            engine="none",
            segments=[],
            saved_path=None,
            error=self.last_error,
            preprocessed=preprocessed,
        )

    def _extract_with_dual_pipeline(self, image_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
        if self._dual_unavailable:
            return "", []
        if not image_bytes:
            return "", []

        try:
            self._ensure_dual_models()
        except Exception as e:
            self._dual_unavailable = True
            self.last_error = f"双引擎初始化失败: {type(e).__name__}: {e}"
            return "", []

        boxes = segment_formula_regions(image_bytes=image_bytes, max_segments=120)
        if not boxes:
            return "", []

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return "", []

        segments: list[dict[str, Any]] = []
        for x, y, w, h in boxes:
            crop = image.crop((x, y, x + w, y + h))
            text, seg_type, score = self._recognize_crop_dual(crop)
            if not text:
                continue
            segments.append(
                {
                    "index": len(segments) + 1,
                    "text": text,
                    "type": seg_type,
                    "bbox": [x, y, x + w, y + h],
                    "score": float(score),
                }
            )

        if not segments:
            return "", []
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        for idx, s in enumerate(segments, start=1):
            s["index"] = idx
        text = "\n".join(seg["text"] for seg in segments if seg["text"].strip()).strip()
        return text, segments

    def _ensure_dual_models(self) -> None:
        cache_root = self._cache_dir
        p2t_home = cache_root / "pix2text"
        cnocr_home = cache_root / "cnocr"
        cnstd_home = cache_root / "cnstd"
        mpl_home = cache_root / "mplconfig"
        for p in [p2t_home, cnocr_home, cnstd_home, mpl_home]:
            p.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("PIX2TEXT_HOME", str(p2t_home))
        os.environ.setdefault("CNOCR_HOME", str(cnocr_home))
        os.environ.setdefault("CNSTD_HOME", str(cnstd_home))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_home))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

        if self._cnocr is None:
            from cnocr import CnOcr  # type: ignore

            self._cnocr = CnOcr(
                det_model_name="naive_det",
                context="cpu",
                rec_root=cnocr_home,
                det_root=cnstd_home,
                rec_more_configs={"ort_providers": ["CPUExecutionProvider"]},
            )
        if self._latex_ocr is None:
            from pix2text.latex_ocr import LatexOCR  # type: ignore

            self._latex_ocr = LatexOCR(
                model_backend="onnx",
                device="cpu",
                root=p2t_home,
                more_model_configs={"provider": "CPUExecutionProvider", "use_cache": False},
            )

    def _recognize_crop_dual(self, crop: Image.Image) -> tuple[str, str, float]:
        text_out = {"text": "", "score": 0.0}
        formula_out = {"text": "", "score": 0.0}

        try:
            text_out = self._cnocr.ocr_for_single_line(crop)  # type: ignore
        except Exception:
            text_out = {"text": "", "score": 0.0}
        try:
            formula_out = self._latex_ocr.recognize(crop)  # type: ignore
        except Exception:
            formula_out = {"text": "", "score": 0.0}

        text_s = str(text_out.get("text", "") or "").strip()
        text_score = float(text_out.get("score", 0.0) or 0.0)
        formula_s = str(formula_out.get("text", "") or "").strip()
        formula_score = float(formula_out.get("score", 0.0) or 0.0)

        # Noise guarding
        if text_s and len(text_s) > 80 and text_score < 0.45:
            text_s = ""
        if formula_s and self._is_likely_gibberish(formula_s):
            formula_s = ""
            formula_score = 0.0

        if not text_s and not formula_s:
            return "", "UNKNOWN", 0.0

        text_has_zh = bool(re.search(r"[\u4e00-\u9fff]", text_s))
        formula_has_zh = bool(re.search(r"[\u4e00-\u9fff]", formula_s))
        formula_mathy = any(tok in formula_s for tok in ["\\int", "\\frac", "\\sin", "\\cos", "^", "_", "="])
        text_mathy = bool(re.search(r"[=+\-*/^_]", text_s))

        # Strongly prefer text OCR when Chinese chars are detected.
        if text_has_zh and (text_score >= 0.22 or not formula_mathy):
            return text_s, "TEXT", text_score
        if formula_has_zh and not formula_mathy:
            formula_s = ""
            formula_score = 0.0

        if text_has_zh and text_score >= 0.15:
            return text_s, "TEXT", text_score
        if formula_mathy and formula_score >= text_score * 0.9:
            return formula_s, "FORMULA", formula_score
        if text_mathy and text_score >= formula_score:
            return text_s, "FORMULA", text_score
        if formula_score > text_score + 0.08:
            return formula_s, "FORMULA", formula_score
        return text_s if text_s else formula_s, "TEXT" if text_s else "FORMULA", max(text_score, formula_score)

    def _extract_with_pix2text(self, image_bytes: bytes) -> tuple[str, list[dict[str, Any]]]:
        if self._p2t_unavailable:
            return "", []

        try:
            if self._p2t is None:
                pix2text_home = self._cache_dir / "pix2text"
                mpl_home = self._cache_dir / "mplconfig"
                hf_home = self._cache_dir / "hf_home"
                font_home = self._cache_dir / "fontconfig"
                ultralytics_home = self._cache_dir / "ultralytics"
                for p in [pix2text_home, mpl_home, hf_home, font_home, ultralytics_home]:
                    p.mkdir(parents=True, exist_ok=True)

                os.environ.setdefault("PIX2TEXT_HOME", str(pix2text_home))
                os.environ.setdefault("PIX2TEXT_DOWNLOAD_SOURCE", "CN")
                os.environ.setdefault("MPLCONFIGDIR", str(mpl_home))
                os.environ.setdefault("HF_HOME", str(hf_home))
                os.environ.setdefault("XDG_CACHE_HOME", str(self._cache_dir))
                os.environ.setdefault("FONTCONFIG_PATH", str(font_home))
                os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_home))
                from pix2text import Pix2Text  # type: ignore

                # Force CPU to avoid CoreMLExecutionProvider dynamic-shape failures on macOS.
                total_configs = {
                    "text_formula": {
                        "formula": {
                            "more_model_configs": {
                                "provider": "CPUExecutionProvider",
                                "use_cache": False,
                            }
                        },
                        "text": {
                            "context": "cpu",
                            "ort_providers": ["CPUExecutionProvider"],
                        },
                        "mfd": {"device": "cpu"},
                    },
                    "layout": {"device": "cpu"},
                }
                self._p2t = Pix2Text.from_config(
                    total_configs=total_configs,
                    enable_formula=True,
                    enable_table=False,
                    device="cpu",
                )
        except Exception as e:
            self._p2t_unavailable = True
            self.last_error = f"Pix2Text 初始化失败: {type(e).__name__}: {e}"
            return "", []

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = self._p2t.recognize(image, file_type="page")
            segments = self._normalize_p2t_result(result)
            lines = [s["text"] for s in segments if s.get("text")]
            return "\n".join(lines).strip(), segments
        except Exception as e:
            self.last_error = f"Pix2Text 识别失败: {type(e).__name__}: {e}"
            return "", []

    def _normalize_p2t_result(self, result: Any) -> list[dict[str, Any]]:
        elements: list[Any] = []
        pages = getattr(result, "pages", None)
        if pages is not None:
            for page in pages:
                elements.extend(getattr(page, "elements", []))
        elif hasattr(result, "elements"):
            elements = list(getattr(result, "elements", []))
        elif isinstance(result, list):
            for item in result:
                if hasattr(item, "elements"):
                    elements.extend(getattr(item, "elements", []))

        segments: list[dict[str, Any]] = []
        for i, ele in enumerate(elements, start=1):
            text = str(getattr(ele, "text", "") or "").strip()
            if not text:
                continue
            box = list(getattr(ele, "box", []) or [])
            if len(box) == 4:
                box = [int(v) for v in box]
            else:
                box = [0, 0, 0, 0]
            ele_type = getattr(getattr(ele, "type", None), "name", None) or str(getattr(ele, "type", "UNKNOWN"))
            score = float(getattr(ele, "score", 0.0) or 0.0)
            segments.append(
                {
                    "index": i,
                    "text": text,
                    "type": ele_type,
                    "bbox": box,
                    "score": score,
                }
            )

        # Ensure stable reading order by bbox when available.
        segments.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        for idx, seg in enumerate(segments, start=1):
            seg["index"] = idx
        return segments

    def _segments_from_text(self, text: str) -> list[dict[str, Any]]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [
            {
                "index": i,
                "text": ln,
                "type": "LINE",
                "bbox": [0, 0, 0, 0],
                "score": 0.0,
            }
            for i, ln in enumerate(lines, start=1)
        ]

    def _save_ocr_run(self, engine: str, raw_text: str, segments: list[dict[str, Any]]) -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        out_path = self._output_dir / f"{run_id}.json"
        payload = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "engine": engine,
            "ocr_text": raw_text,
            "segments": segments,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(out_path)

    def _extract_with_layout_pipeline(self, image_bytes: bytes) -> str | None:
        if not image_bytes:
            return None
        boxes = segment_formula_regions(image_bytes=image_bytes, max_segments=90)
        if len(boxes) < 2:
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

        lines: list[str] = []
        for x, y, w, h in boxes:
            crop = image.crop((x, y, x + w, y + h))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            chunk_bytes = buf.getvalue()

            text = self._extract_with_rapid_latex_ocr(chunk_bytes) or self._extract_with_pix2tex(chunk_bytes)
            if not text:
                continue
            normalized = self._normalize_chunk(text)
            if self._is_likely_gibberish(normalized):
                continue
            if not self._looks_like_math(normalized):
                continue
            if lines and normalized == lines[-1]:
                continue
            lines.append(normalized)

        if len(lines) < 2:
            return None
        return "\n".join(lines)

    def _normalize_chunk(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _looks_like_math(self, text: str) -> bool:
        if len(text) < 2:
            return False
        math_tokens = [
            "\\int",
            "\\frac",
            "\\sin",
            "\\cos",
            "\\ln",
            "\\pi",
            "\\sqrt",
            "=",
            "^",
            "+",
            "-",
            "/",
        ]
        if any(tok in text for tok in math_tokens):
            return True
        return bool(re.search(r"\d", text))

    def _is_likely_gibberish(self, text: str) -> bool:
        if text.count("\\vdots") >= 2:
            return True
        if text.count("\\varphi") >= 3:
            return True
        if text.count("\\theta") >= 3:
            return True
        if "\\mathrm" in text and not re.search(r"[0-9=+\-*/^]", text):
            return True
        if text.startswith("\\scriptstyle") and not re.search(r"[0-9=+\-*/^]", text):
            return True
        if len(text) > 140 and not re.search(r"[=0-9]", text):
            return True
        return False

    def _extract_with_rapid_latex_ocr(self, image_bytes: bytes) -> str | None:
        if self._rapid_unavailable:
            return None
        try:
            if self._rapid_model is None:
                from rapid_latex_ocr import LaTeXOCR as RapidLaTeXOCR  # type: ignore

                self._rapid_model = RapidLaTeXOCR()
        except Exception:
            self._rapid_unavailable = True
            return None

        try:
            result = self._rapid_model(image_bytes)
            text = ""
            if isinstance(result, str):
                text = result
            elif isinstance(result, (tuple, list)) and result:
                text = str(result[0])
            text = text.strip()
            if not text or self._is_likely_gibberish(text):
                return None
            return text
        except Exception:
            return None

    def _extract_with_pix2tex(self, image_bytes: bytes) -> str | None:
        if self._pix2tex_unavailable:
            return None
        try:
            if self._pix2tex_model is None:
                from pix2tex.cli import LatexOCR  # type: ignore

                self._pix2tex_model = LatexOCR()
        except Exception:
            self._pix2tex_unavailable = True
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = self._pix2tex_model(image)
            text = (result or "").strip()
            if not text or self._is_likely_gibberish(text):
                return None
            return text
        except Exception:
            return None
