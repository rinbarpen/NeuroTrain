"""HTML reporting utilities for NeuroTrain.

This module renders rich HTML reports for training / inference runs by
combining metrics produced by the analyzer suite, visual artefacts (png,
svg, etc.) and monitoring JSON logs.  Optionally, it can convert the
HTML document into a PDF file.

Key features
------------
* Render HTML using Jinja2 templates (supports custom template override)
* Embed static artefacts (plots, tables, JSON summaries)
* Inline small images as Base64 or link to external files
* Convert HTML to PDF with ``weasyprint`` or ``pdfkit`` (if available)

Example
-------

```python
from pathlib import Path
from tools.reporting import HTMLReportGenerator

generator = HTMLReportGenerator(template_dir=Path("docs/templates"))

report_path = generator.render(
    output_dir=Path("runs/example/report"),
    context={
        "title": "NeuroTrain Experiment Report",
        "summary": summary_dict,
        "metrics": metrics_dict,
        "charts": list_of_png_paths,
        "monitor": monitor_json_path,
    },
    html_filename="report.html",
    convert_to_pdf=True,
)

print(f"Report generated at {report_path}")
```
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # Optional dependency
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Environment = None  # type: ignore

# Optional PDF backends -----------------------------------------------------
try:
    import weasyprint  # type: ignore

    _WEASY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _WEASY_AVAILABLE = False

try:
    import pdfkit  # type: ignore

    _PDFKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _PDFKIT_AVAILABLE = False


class ReportRenderingError(RuntimeError):
    """Raised when report rendering fails."""


@dataclass
class HTMLReportConfig:
    """Configuration for the HTML report generator."""

    template_name: str = "report.html.j2"
    template_dir: Optional[Path] = None
    static_dir_name: str = "static"
    embed_images: bool = True
    embed_threshold_kb: int = 512  # Inline small images (< threshold)
    pdf_backend: str = "auto"  # "auto", "weasyprint", "pdfkit", "none"
    pdf_filename: str = "report.pdf"
    extra_context: Dict[str, Any] = field(default_factory=dict)


class HTMLReportGenerator:
    """Generate HTML (and optional PDF) reports for NeuroTrain runs."""

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        config: Optional[HTMLReportConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or HTMLReportConfig()
        if template_dir:
            self.config.template_dir = template_dir

        self.logger = logger or logging.getLogger("HTMLReportGenerator")
        self.logger.setLevel(logging.INFO)

        if Environment is None:  # pragma: no cover
            raise ReportRenderingError(
                "Jinja2 is required for HTML reporting. Install it via 'pip install jinja2'."
            )

        template_dir = self._resolve_template_dir()
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "j2"]),
        )

    # ------------------------------------------------------------------
    def render(
        self,
        output_dir: Path,
        context: Dict[str, Any],
        html_filename: str = "report.html",
        convert_to_pdf: bool = False,
    ) -> Path:
        """Render an HTML report and optionally generate a PDF copy.

        Parameters
        ----------
        output_dir:
            Directory where the report will be saved.
        context:
            Data passed to the Jinja2 template. May include keys such as
            ``summary``, ``metrics``, ``charts`` (list of image paths), etc.
        html_filename:
            Name of the generated HTML file.
        convert_to_pdf:
            If True, also produce a PDF (requires optional backend).

        Returns
        -------
        Path to the generated HTML report.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        static_dir = output_dir / self.config.static_dir_name
        static_dir.mkdir(parents=True, exist_ok=True)

        template = self.env.get_template(self.config.template_name)

        full_context = dict(self.config.extra_context)
        full_context.update(self._prepare_context(context, static_dir))

        html_text = template.render(**full_context)
        html_path = output_dir / html_filename
        html_path.write_text(html_text, encoding="utf-8")
        self.logger.info("HTML report written to %s", html_path)

        if convert_to_pdf:
            backend = self._choose_pdf_backend(self.config.pdf_backend)
            if backend == "weasyprint":
                self._convert_with_weasyprint(html_path, output_dir / self.config.pdf_filename)
            elif backend == "pdfkit":
                self._convert_with_pdfkit(html_path, output_dir / self.config.pdf_filename)
            else:
                self.logger.warning("No PDF backend available; skipping PDF generation")

        return html_path

    # ------------------------------------------------------------------
    def _prepare_context(self, context: Dict[str, Any], static_dir: Path) -> Dict[str, Any]:
        prepared = dict(context)

        # Resolve monitor JSON if provided as path
        monitor_info = prepared.get("monitor")
        if isinstance(monitor_info, (str, Path)):
            try:
                prepared["monitor_data"] = json.loads(Path(monitor_info).read_text("utf-8"))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.warning("Failed to load monitor JSON: %s", exc)
                prepared["monitor_data"] = None
        else:
            prepared["monitor_data"] = monitor_info

        # Handle charts/images
        charts: Iterable[Any] = prepared.get("charts", [])
        processed_charts = []
        for chart in charts:
            processed_charts.append(
                self._process_image(chart, static_dir)
            )
        prepared["charts"] = processed_charts

        # Support embedding additional images via "images" context key
        images: Iterable[Any] = prepared.get("images", [])
        prepared["images"] = [self._process_image(img, static_dir) for img in images]

        return prepared

    # ------------------------------------------------------------------
    def _process_image(self, image_entry: Any, static_dir: Path) -> Dict[str, Any]:
        if isinstance(image_entry, dict):  # already structured
            path = image_entry.get("path")
            caption = image_entry.get("caption")
        else:
            path = image_entry
            caption = None

        if path is None:
            return {"type": "unknown", "caption": caption}

        path = Path(path)
        if not path.exists():
            self.logger.warning("Image not found: %s", path)
            return {"type": "missing", "caption": caption, "path": str(path)}

        if self.config.embed_images and path.stat().st_size < self.config.embed_threshold_kb * 1024:
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            mime = self._guess_mime(path.suffix)
            return {"type": "inline", "data_uri": f"data:{mime};base64,{encoded}", "caption": caption}

        # Copy file into static dir for referencing from HTML
        target = static_dir / path.name
        if path.resolve() != target.resolve():
            target.write_bytes(path.read_bytes())
        return {"type": "file", "path": f"{self.config.static_dir_name}/{target.name}", "caption": caption}

    # ------------------------------------------------------------------
    @staticmethod
    def _guess_mime(suffix: str) -> str:
        suffix = suffix.lower()
        if suffix in {".png"}:
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix in {".svg"}:
            return "image/svg+xml"
        return "application/octet-stream"

    # ------------------------------------------------------------------
    def _resolve_template_dir(self) -> Path:
        if self.config.template_dir and self.config.template_dir.exists():
            return self.config.template_dir

        # Fallback to built-in templates directory
        builtin = Path(__file__).parent / "templates"
        if builtin.exists():
            return builtin

        raise ReportRenderingError("No template directory found")

    # ------------------------------------------------------------------
    def _choose_pdf_backend(self, preference: str) -> str:
        preference = preference.lower()
        if preference == "none":
            return "none"
        if preference == "weasyprint" and _WEASY_AVAILABLE:
            return "weasyprint"
        if preference == "pdfkit" and _PDFKIT_AVAILABLE:
            return "pdfkit"
        if preference == "auto":
            if _WEASY_AVAILABLE:
                return "weasyprint"
            if _PDFKIT_AVAILABLE:
                return "pdfkit"
        return "none"

    # ------------------------------------------------------------------
    def _convert_with_weasyprint(self, html_path: Path, pdf_path: Path) -> None:
        if not _WEASY_AVAILABLE:  # pragma: no cover - safety
            raise ReportRenderingError("WeasyPrint not installed")

        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        self.logger.info("PDF generated via WeasyPrint at %s", pdf_path)

    # ------------------------------------------------------------------
    def _convert_with_pdfkit(self, html_path: Path, pdf_path: Path) -> None:
        if not _PDFKIT_AVAILABLE:  # pragma: no cover - safety
            raise ReportRenderingError("pdfkit not installed")

        # pdfkit needs wkhtmltopdf executable available in PATH
        pdfkit.from_file(str(html_path), str(pdf_path))  # type: ignore[arg-type]
        self.logger.info("PDF generated via pdfkit at %s", pdf_path)


# Backward compatibility convenience ---------------------------------------
def render_report(
    context: Dict[str, Any],
    output_dir: Path,
    template_dir: Optional[Path] = None,
    html_filename: str = "report.html",
    convert_to_pdf: bool = True,
) -> Path:
    """Convenience wrapper around :class:`HTMLReportGenerator`."""

    generator = HTMLReportGenerator(template_dir=template_dir)
    return generator.render(
        output_dir=output_dir,
        context=context,
        html_filename=html_filename,
        convert_to_pdf=convert_to_pdf,
    )

