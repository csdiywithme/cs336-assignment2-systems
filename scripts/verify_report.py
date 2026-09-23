"""Render every report page and audit bounds/links using only local resources."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw
import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/assignment2_existing_evidence_report.pdf"
QA = ROOT / "tmp/pdfs/qa" / hashlib.sha256(PDF.read_bytes()).hexdigest()[:12]
POPPLER = Path("/Users/simida/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm")


def main():
    QA.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(PDF)
    prefix = QA / "page"
    subprocess.run([str(POPPLER), "-r", "100", "-png", str(PDF), str(prefix)], check=True)
    bounds = []
    pages = []
    with pdfplumber.open(PDF) as doc:
        for i, page in enumerate(doc.pages, 1):
            chars = page.chars
            for ch in chars:
                if ch["text"].strip() and (ch["x0"] < 3 or ch["x1"] > page.width - 3
                                           or ch["top"] < 3 or ch["bottom"] > page.height - 3):
                    bounds.append({"page": i, "text": ch["text"], "x0": ch["x0"], "x1": ch["x1"],
                                   "top": ch["top"], "bottom": ch["bottom"]})
            pages.append({"page": i, "characters": len(chars), "images": len(page.images)})
    rendered = sorted(QA.glob("page-*.png"))
    if len(rendered) != len(reader.pages):
        raise RuntimeError("Render page count mismatch; inspect previous QA files")
    contacts = []
    for start in range(0, len(rendered), 6):
        contact = Image.new("RGB", (840, 1845), "#E7EDF1")
        draw = ImageDraw.Draw(contact)
        for j, p in enumerate(rendered[start:start + 6]):
            with Image.open(p) as im:
                im.thumbnail((400, 566))
                x, y = 15 + (j % 2) * 420, 28 + (j // 2) * 615
                contact.paste(im, (x, y))
                draw.text((x, y - 17), f"Page {start + j + 1}", fill="#142E41")
        path = QA / f"contact-{start // 6 + 1:02}.png"
        contact.save(path)
        contacts.append(str(path.relative_to(ROOT)))
    bad_links = []
    for source in [ROOT / "output/README.md", ROOT / "output/FULL_REPORT.md", ROOT / "HANDOUT_ANSWERS.md",
                   ROOT / "EXPERIMENT_LOG.md", ROOT / "GPU_REMAINING_TASKS.md"]:
        content = re.sub(r"```[\s\S]*?```", "", source.read_text())
        content = re.sub(r"`[^`]+`", "", content)
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            path = source.parent / target.split("#", 1)[0]
            if not path.exists():
                bad_links.append({"source": str(source.relative_to(ROOT)), "target": target})
    report = {"pdf": str(PDF.relative_to(ROOT)), "sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
              "page_count": len(reader.pages), "rendered_pages": len(rendered), "pages": pages,
              "out_of_page_text": bounds, "missing_local_markdown_links": bad_links,
              "contact_sheets": contacts, "visual_review": "Required after rendering; not inferred from automated checks."}
    (QA / "qa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    verification = ROOT / "output/verification"
    verification.mkdir(parents=True, exist_ok=True)
    (verification / "automated_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "pages"}, indent=2, ensure_ascii=False))
    if bounds or bad_links:
        raise SystemExit("QA issues detected")


if __name__ == "__main__":
    main()
