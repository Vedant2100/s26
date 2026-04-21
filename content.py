#!/usr/bin/env python3
"""
Automated Canvas Course Content Downloader

This script downloads all course materials from both current and past Canvas courses
using the Canvas API. It supports all file types, including `.pdf`, `.R`, `.Rmd`,
`.csv`, `.ipynb`, and others. Files embedded inside pages, modules, or viewer-style
links are automatically identified and downloaded.

Setup and Usage:
1. Install Required Python Packages:
   - requests
   - beautifulsoup4
   - pdfkit

2. Install wkhtmltopdf:
   - This tool is required by pdfkit to convert HTML content (pages, assignments,
     modules) into PDF files.
   - Download and install it from: https://wkhtmltopdf.org/downloads.html
   - Default path after installation will work.

3. Set Environment Variables:
   - Log into Canvas, go to Account > Settings > Approved Integrations, and
     generate a new token.
   - Set CANVAS_API_TOKEN environment variable with your API token.
   - Set CANVAS_DOMAIN environment variable with your Canvas domain
     (e.g., 'https://canvas.pitt.edu').
   - For GitHub Actions, add these as secrets in your repository settings.

4. Run the Script:
   - python canvas_course_downloader.py
   - It will retrieve currently enrolled (active) Canvas courses only
   - Download all available files, linked content, and assignment submissions
   - Convert Canvas-hosted HTML content into PDFs (no `.html` files are saved)
   - Save everything to a structured local folder organized by course

Output:
All downloaded files are saved to a local directory named `canvas_all_content`,
with one subfolder per course. Original filenames and extensions are preserved.

Folder Structure:
canvas_all_content/
├── Course Name A/
│   ├── lecture1.pdf
│   ├── page - Syllabus.pdf
│   ├── assignment - Essay.pdf
│   ├── module - Week 1 Overview.pdf
│   └── submission - final_essay.pdf
├── Course Name B/
│   └── ...

Notes:
- Most module and assignment PDFs may appear blank. This is expected behavior:
  - Modules are often used as containers for linked content rather than
    standalone descriptions.
  - Assignment pages are also frequently blank unless the instructor specifically
    writes assignment details in the Canvas page itself.
  - These files are still processed because they often contain embedded links
    to downloadable materials.
"""

import os
import re
import json
import platform
import shutil
import zipfile
import tempfile
import requests
import subprocess
import pdfkit
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Configuration for wkhtmltopdf (platform-agnostic)
# Try to find wkhtmltopdf automatically
wkhtmltopdf_path = shutil.which("wkhtmltopdf")
if wkhtmltopdf_path:
    pdfkit_config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
else:
    # Fallback paths for different platforms
    if platform.system() == "Windows":
        pdfkit_config = pdfkit.configuration(
            wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
        )
    elif platform.system() == "Darwin":  # macOS
        # Common macOS installation paths
        possible_paths = [
            "/usr/local/bin/wkhtmltopdf",
            "/opt/homebrew/bin/wkhtmltopdf",
            "/usr/bin/wkhtmltopdf",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pdfkit_config = pdfkit.configuration(wkhtmltopdf=path)
                break
        else:
            # If not found, let pdfkit try to find it automatically
            pdfkit_config = pdfkit.configuration()
    else:
        # Linux or other Unix-like systems
        pdfkit_config = pdfkit.configuration()


# Configuration from config.json
# Load configuration file, with fallback to environment variables
CONFIG = {}
try:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            CONFIG = json.load(f)
        print(f"✓ Loaded configuration from {config_path}")
except Exception as e:
    print(f"⚠️  Could not load config.json: {e}")
    CONFIG = {}


# Get Canvas API token and domain from environment variables
# Strip whitespace to avoid issues with GitHub Actions secret injection
CANVAS_API_TOKEN = (os.getenv("CANVAS_API_TOKEN") or "").strip()
CANVAS_DOMAIN = (os.getenv("CANVAS_DOMAIN") or "").strip()

# Don't raise at import time; instead record missing creds and let main() decide.
MISSING_CANVAS_CREDS = False
if not CANVAS_API_TOKEN:
    print("⚠️  CANVAS_API_TOKEN is not set. Some operations will fail without it.")
    MISSING_CANVAS_CREDS = True

if not CANVAS_DOMAIN:
    print("⚠️  CANVAS_DOMAIN is not set. Some operations will fail without it.")
    MISSING_CANVAS_CREDS = True


def ensure_canvas_creds():
    """Raise a clear error if required Canvas credentials are missing."""
    if MISSING_CANVAS_CREDS:
        raise RuntimeError(
            "Canvas API credentials are missing. Set CANVAS_API_TOKEN and CANVAS_DOMAIN in the environment or repo secrets."
        )


BASE_API_URL = f"{CANVAS_DOMAIN}/api/v1"
HEADERS = {"Authorization": f"Bearer {CANVAS_API_TOKEN}"}

# Determine where to save files - use config or repo directory if in GitHub Actions, otherwise Downloads
config_output = CONFIG.get("output", {})
base_dir_config = config_output.get("base_directory", "canvas_all_content")

if os.getenv("GITHUB_WORKSPACE"):
    # Running in GitHub Actions - save to repo
    DOWNLOADS_BASE = os.path.join(os.getenv("GITHUB_WORKSPACE"), base_dir_config)
else:
    # Running locally - save to Downloads
    DOWNLOADS_BASE = os.path.join(os.path.expanduser("~"), "Downloads", base_dir_config)

# Whether to download student submissions (from config, with env override)
config_sync = CONFIG.get("sync", {})
download_submissions_config = config_sync.get("download_submissions", False)
DOWNLOAD_SUBMISSIONS = (
    os.getenv("DOWNLOAD_SUBMISSIONS", "false").lower() == "true"
    or download_submissions_config
)

# Internal git commit/push is disabled by default because the GitHub workflow
# already handles commit/push in a dedicated step.
INTERNAL_GIT_COMMIT = os.getenv("INTERNAL_GIT_COMMIT", "false").lower() == "true"

downloaded_file_urls = set()
# Maximum file size to save into the repo/workspace (bytes). Files larger than
# this will be skipped to avoid committing very large binaries. Can be overridden
# by setting the MAX_SAVE_BYTES environment variable in the workflow.
MAX_SAVE_BYTES = int(os.getenv("MAX_SAVE_BYTES", str(100 * 1024 * 1024)))
# Video file extensions to skip saving into the repo/workspace (to avoid large binaries)
VIDEO_EXTS = {".mp4", ".mpg", ".mpeg", ".mov", ".avi", ".mkv", ".webm"}

# Global flag set per-course when we should only save PDFs for that course
CURRENT_COURSE_ONLY_PDF = False
# Video file extensions to skip saving into the repo/workspace (to avoid large binaries)
VIDEO_EXTS = {".mp4", ".mpg", ".mpeg", ".mov", ".avi", ".mkv", ".webm"}


def make_safe(name):
    """Sanitize filename by removing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def extract_and_save_zip(zip_content, course_folder, zip_filename):
    """Extract a zip file and save all its contents to the course folder.

    Args:
        zip_content: The binary content of the zip file
        course_folder: The destination folder for extracted files
        zip_filename: Original zip filename (for logging)

    Returns:
        List of extracted file paths
    """
    extracted_files = []
    try:
        # Create a temporary file to write the zip content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_file.write(zip_content)
            tmp_path = tmp_file.name

        # Extract the zip file
        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                # Skip directories and hidden/system files
                if (
                    member.endswith("/")
                    or member.startswith("__MACOSX")
                    or member.startswith(".")
                ):
                    continue

                # Get just the filename (flatten directory structure)
                original_basename = os.path.basename(member)
                if not original_basename:
                    continue

                safe_name = make_safe(original_basename)
                dest_path = os.path.join(course_folder, safe_name)

                # If a file with the same name exists, check if content matches;
                # if identical, skip creating a duplicate. Otherwise add suffix.
                base, ext = os.path.splitext(safe_name)
                counter = 1
                if os.path.exists(dest_path):
                    try:
                        existing = open(dest_path, "rb").read()
                        if existing == zip_ref.read(member):
                            extracted_files.append(dest_path)
                            print(f"      ⚪ Skipped duplicate (identical): {safe_name}")
                            continue
                    except Exception:
                        # If any error comparing, fall back to creating a new name
                        pass

                while os.path.exists(dest_path):
                    safe_name = f"{base}_{counter}{ext}"
                    dest_path = os.path.join(course_folder, safe_name)
                    counter += 1

                # Extract the file content and save it
                try:
                    file_content = zip_ref.read(member)

                    # Skip PDFs that are placeholder 'assignment - ...' files
                    lower_name = safe_name.lower()
                    if lower_name.endswith('.pdf') and lower_name.startswith('assignment - '):
                        skipped_dir = os.path.join(course_folder, "__skipped_assignment_placeholders__")
                        os.makedirs(skipped_dir, exist_ok=True)
                        with open(os.path.join(skipped_dir, f"{make_safe(safe_name)}.meta.txt"), "w", encoding="utf-8") as m:
                            m.write(f"original_name: {safe_name}\nreason: assignment_placeholder_skipped_from_zip\n")
                        print(f"    ⛔ Skipped placeholder assignment PDF in zip: {safe_name}")
                        continue
                    with open(dest_path, "wb") as f:
                        f.write(file_content)
                    extracted_files.append(dest_path)
                    print(f"      📦 Extracted from zip: {safe_name}")
                except Exception as e:
                    print(f"      ⚠️  Error extracting {member}: {e}")

        # Clean up temporary file
        os.unlink(tmp_path)
        print(f"    ✅ Extracted {len(extracted_files)} files from {zip_filename}")

    except zipfile.BadZipFile:
        print(f"    ⚠️  {zip_filename} is not a valid zip file, saving as-is")
        return None  # Signal that it should be saved as regular file
    except Exception as e:
        print(f"    ❌ Error extracting zip {zip_filename}: {e}")
        return None  # Signal that it should be saved as regular file

    return extracted_files


def _find_libreoffice():
    """Return path to a LibreOffice/soffice binary or None if not found."""
    for name in ("soffice", "libreoffice", "soffice.bin"):
        path = shutil.which(name)
        if path:
            return path
    return None


def convert_pptx_to_pdf(pptx_path, out_dir=None):
    """Convert a .pptx file to PDF using LibreOffice in headless mode.

    Returns path to the generated PDF on success, or None on failure.
    """
    soffice = _find_libreoffice()
    if not soffice:
        print("    ⚠️  LibreOffice/soffice not found; cannot convert PPTX to PDF")
        return None

    out_dir = out_dir or os.path.dirname(pptx_path)
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, pptx_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base, _ = os.path.splitext(os.path.basename(pptx_path))
        pdf_path = os.path.join(out_dir, f"{base}.pdf")
        if os.path.exists(pdf_path):
            print(f"    ✅ Converted PPTX to PDF: {os.path.basename(pdf_path)}")
            return pdf_path
        else:
            print(f"    ⚠️  Conversion finished but PDF not found for {pptx_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"    ❌ LibreOffice failed converting {pptx_path}: {e}")
        return None


def download_dropbox_folder(shared_url, course_folder):
    """Download a shared Dropbox folder (shared_url) as a zip and extract it.

    The function will attempt to force a direct download (dl=1) which returns a
    zip archive of the folder contents. Extracted files are saved into
    `course_folder`. Any `.pptx` files found will be converted to PDF (if
    LibreOffice is available) and the original `.pptx` removed so only PDFs
    remain in the repo as requested.
    """
    os.makedirs(course_folder, exist_ok=True)

    # Force direct-download (Dropbox uses dl=1 for raw download)
    if "dl=1" in shared_url:
        dl_url = shared_url
    else:
        if "?" in shared_url:
            # replace existing dl param if present, otherwise append
            if "dl=" in shared_url:
                dl_url = re.sub(r"dl=\d", "dl=1", shared_url)
            else:
                dl_url = shared_url + "&dl=1"
        else:
            dl_url = shared_url + "?dl=1"

    print(f"  🔗 Fetching Dropbox folder: {dl_url}")
    try:
        r = requests.get(dl_url, stream=True, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"    ❌ Error downloading Dropbox link: {e}")
        return

    content = r.content

    # Try to extract as zip archive
    extracted = extract_and_save_zip(content, course_folder, "dropbox_folder.zip")
    if extracted is None:
        # If extraction failed, save the raw content as a single file
        dest = os.path.join(course_folder, "dropbox_folder.zip")
        with open(dest, "wb") as f:
            f.write(content)
        print(f"    Saved Dropbox content as {dest}")
        return

    # Convert any PPTX files to PDF and remove the originals
    for root, _, files in os.walk(course_folder):
        for fname in files:
            if fname.lower().endswith(".pptx"):
                pptx_path = os.path.join(root, fname)
                pdf_path = convert_pptx_to_pdf(pptx_path, out_dir=root)
                if pdf_path:
                    try:
                        os.remove(pptx_path)
                        print(f"    🧹 Removed source PPTX: {fname}")
                    except Exception as e:
                        print(f"    ⚠️  Could not remove {pptx_path}: {e}")


def safe_paginate(url):
    """Safely paginate through API results."""
    results = []
    try:
        while url:
            r = requests.get(url, headers=HEADERS)
            if r.status_code in [403, 404]:
                print(f"    Skipping ({r.status_code} error): {url}")
                return []
            r.raise_for_status()
            results.extend(r.json())
            url = r.links.get("next", {}).get("url")
        return results
    except Exception as e:
        print(f"    Error during pagination: {e}")
        return []


def save_html_as_pdf(folder, name, html_content):
    """Convert HTML content to PDF and save it."""
    # Skip generated PDFs for assignment placeholders (names like 'assignment - ...')
    if name and name.strip().lower().startswith('assignment - '):
        skipped_dir = os.path.join(folder, "__skipped_assignment_placeholders__")
        os.makedirs(skipped_dir, exist_ok=True)
        with open(os.path.join(skipped_dir, f"{make_safe(name)}.meta.txt"), "w", encoding="utf-8") as m:
            m.write(f"generated_name: {name}\nreason: assignment_placeholder_skipped_from_html\n")
        print(f"    ⛔ Skipped generating placeholder assignment PDF: {name}")
        return

    safe_name = make_safe(name)
    pdf_path = os.path.join(folder, f"{safe_name}.pdf")
    try:
        pdfkit.from_string(html_content, pdf_path, configuration=pdfkit_config)
        print(f"    Saved PDF: {safe_name}.pdf")
    except Exception as e:
        print(f"    Error converting {safe_name} to PDF: {e}")


def save_markdown(folder, name, markdown_content):
    """Save markdown content to a .md file."""
    safe_name = make_safe(name)
    md_path = os.path.join(folder, f"{safe_name}.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"    Saved Markdown: {safe_name}.md")
    except Exception as e:
        print(f"    Error saving {safe_name}.md: {e}")


def save_or_unzip(content, folder, filename):
    """Save content to file, or unzip if it's a zip file."""
    if filename.lower().endswith(".zip"):
        print(f"    📦 Detected ZIP file: {filename}, extracting...")
        result = extract_and_save_zip(content, folder, filename)
        if result is not None:
            return  # Successfully extracted

    # Not a zip or extraction failed - save as regular file
    # Avoid saving video files (user requested) and huge files into the repo/workspace
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    # If this course is configured to only keep PDFs, skip non-pdfs
    if CURRENT_COURSE_ONLY_PDF and ext != ".pdf":
        print(f"    ⛔ Skipping non-PDF for course-only-PDF mode: {filename}")
        skipped_dir = os.path.join(folder, "__skipped_nonpdfs__")
        os.makedirs(skipped_dir, exist_ok=True)
        with open(os.path.join(skipped_dir, f"{make_safe(filename)}.meta.txt"), "w", encoding="utf-8") as m:
            m.write(f"original_name: {filename}\nreason: course_only_pdf\n")
        return

    if ext in VIDEO_EXTS:
        print(f"    ⛔ Skipping video file: {filename}")
        skipped_dir = os.path.join(folder, "__skipped_videos__")
        os.makedirs(skipped_dir, exist_ok=True)
        with open(os.path.join(skipped_dir, f"{make_safe(filename)}.meta.txt"), "w", encoding="utf-8") as m:
            m.write(f"original_name: {filename}\nreason: video extension\n")
        return

    # Skip placeholder PDFs that start with 'assignment - '
    if ext == ".pdf" and filename.lower().startswith('assignment - '):
        skipped_dir = os.path.join(folder, "__skipped_assignment_placeholders__")
        os.makedirs(skipped_dir, exist_ok=True)
        with open(os.path.join(skipped_dir, f"{make_safe(filename)}.meta.txt"), "w", encoding="utf-8") as m:
            m.write(f"original_name: {filename}\nreason: assignment_placeholder_skipped_from_download\n")
        print(f"    ⛔ Skipped placeholder assignment PDF: {filename}")
        return

    size = len(content) if content is not None else 0
    if size and size > MAX_SAVE_BYTES:
        print(f"    ⚠️  Skipping {filename}: size {size} bytes exceeds MAX_SAVE_BYTES ({MAX_SAVE_BYTES})")
        # Save metadata about skipped file for later inspection
        skipped_dir = os.path.join(folder, "__skipped_large_files__")
        os.makedirs(skipped_dir, exist_ok=True)
        with open(os.path.join(skipped_dir, f"{make_safe(filename)}.meta.txt"), "w", encoding="utf-8") as m:
            m.write(f"original_name: {filename}\nsize_bytes: {size}\n")
        return

    file_path = os.path.join(folder, filename)
    # If file already exists and is identical, skip writing to avoid duplicates
    if os.path.exists(file_path):
        try:
            existing = open(file_path, "rb").read()
            if existing == content:
                print(f"    ⚪ Skipped saving; identical file already exists: {filename}")
                return
        except Exception:
            pass

    with open(file_path, "wb") as f:
        f.write(content)
    # If this is a PPTX, try to convert it to PDF and remove the original PPTX
    try:
        if file_path.lower().endswith(".pptx"):
            pdf_path = convert_pptx_to_pdf(file_path, out_dir=folder)
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(file_path)
                    print(f"    🧹 Removed source PPTX after conversion: {filename}")
                except Exception as e:
                    print(f"    ⚠️  Could not remove PPTX {file_path}: {e}")
    except Exception:
        # Conversion failures should not stop the downloader
        pass


def download_canvas_file_by_id(file_id, course_folder):
    """Download a Canvas file by its file ID."""
    try:
        meta = requests.get(f"{BASE_API_URL}/files/{file_id}", headers=HEADERS)
        meta.raise_for_status()
        file_data = meta.json()
        download_url = file_data["url"]
        filename = make_safe(file_data["display_name"])

        if download_url in downloaded_file_urls:
            return

        r = requests.get(download_url, headers=HEADERS)
        r.raise_for_status()
        save_or_unzip(r.content, course_folder, filename)

        downloaded_file_urls.add(download_url)
        print(f"    ✅ Downloaded file from API: {filename}")
    except Exception as e:
        print(f"    ❌ Error downloading file ID {file_id}: {e}")


def extract_and_download_linked_files(html, course_folder):
    """Extract file IDs from HTML and download the associated files."""
    if not html:
        return

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(["a", "iframe"], href=True) + soup.find_all(
        ["a", "iframe"], src=True
    ):
        href = tag.get("href") or tag.get("src")
        if href:
            match = re.search(r"/files/(\d+)", href)
            if match:
                file_id = match.group(1)
                download_canvas_file_by_id(file_id, course_folder)

    for script in soup.find_all("script"):
        if script.string:
            matches = re.findall(r"/files/(\d+)", script.string)
            for file_id in set(matches):
                download_canvas_file_by_id(file_id, course_folder)


def main():
    """Main workflow to download all course content."""
    # Ensure the downloads directory exists
    os.makedirs(DOWNLOADS_BASE, exist_ok=True)

    # Ensure credentials are present; raise a friendly message if not.
    try:
        ensure_canvas_creds()
    except RuntimeError as e:
        print(f"❌ {e}")
        print(
            "Exiting. If you're running in GitHub Actions, ensure the secrets are set in the repository and the workflow step has the correct env."
        )
        return

    print("Fetching your Canvas courses...")

    # Only fetch currently enrolled (active) courses
    courses = safe_paginate(
        f"{BASE_API_URL}/courses?per_page=100&enrollment_state=active"
    )

    for course in courses:
        course_id = course["id"]
        course_name = make_safe(course.get("name") or f"course_{course_id}")
        print(f"\nCourse: {course_name}")
        course_folder = os.path.join(DOWNLOADS_BASE, course_name)
        os.makedirs(course_folder, exist_ok=True)

        # Always skip Deep Learning course (explicit user request)
        if "DEEP LEARNING" in course_name.upper():
            print(f"  ⛔ Skipping course (configured to skip Deep Learning): {course_name}")
            continue

        # Skip SHAPE and Stay TA Ready courses per user request
        upper_name = course_name.upper()
        if "SHAPE" in upper_name or "STAY TA READY" in upper_name:
            print(f"  ⛔ Skipping course (configured to skip): {course_name}")
            continue

        # If this course should only keep PDFs (Advanced Computer Vision), set flag
        global CURRENT_COURSE_ONLY_PDF
        if "ADVANCED COMPUTER VISION" in upper_name:
            CURRENT_COURSE_ONLY_PDF = True
        else:
            CURRENT_COURSE_ONLY_PDF = False

        # If this course should be sourced from a Dropbox shared-folder instead
        # of Canvas, the URL can be provided via the `DROPBOX_ML_URL` env var
        # or via a mapping in `config.json` under the `dropbox_urls` key.
        dropbox_url = None
        mapping = CONFIG.get("dropbox_urls", {})
        if course_name in mapping:
            dropbox_url = mapping[course_name]
        else:
            # Allow case-insensitive or partial matches from config keys
            for k, v in mapping.items():
                if k and (k.strip().upper() in course_name.upper() or course_name.upper() in k.strip().upper()):
                    dropbox_url = v
                    break

        # Env var override (explicit for Machine Learning course)
        env_dropbox = os.getenv("DROPBOX_ML_URL", "").strip()
        if env_dropbox:
            dropbox_url = env_dropbox

        # If this looks like the Machine Learning course and a dropbox URL is set,
        # fetch the shared folder and skip Canvas downloading for this course.
        if dropbox_url and "MACHINE LEARNING" in course_name.upper():
            print("  ⚙️  Detected Machine Learning course - fetching from Dropbox instead of Canvas")
            download_dropbox_folder(dropbox_url, course_folder)
            # After fetching from Dropbox, skip the Canvas-based download flow
            continue

        # Download announcements
        print("  Downloading announcements...")
        announcements_folder = os.path.join(course_folder, "announcements")
        os.makedirs(announcements_folder, exist_ok=True)
        try:
            announcements = safe_paginate(f"{BASE_API_URL}/courses/{course_id}/announcements?per_page=100")
            for ann in announcements:
                ann_title = make_safe(ann.get("title", "announcement"))
                ann_filename = f"{ann_title}.md"
                ann_path = os.path.join(announcements_folder, ann_filename)
                ann_body = ann.get("message", "")
                ann_author = ann.get("author", {}).get("display_name", "")
                ann_date = ann.get("posted_at", "")
                md = f"# {ann.get('title', 'Announcement') }\n\n*By: {ann_author}*\n*Posted: {ann_date}*\n\n---\n\n{ann_body}"
                with open(ann_path, "w", encoding="utf-8") as f:
                    f.write(md)
                print(f"    ✅ Saved announcement: {ann_filename}")
        except Exception as e:
            print(f"    Error downloading announcements: {e}")

        # Download discussions
        print("  Downloading discussions...")
        discussions_folder = os.path.join(course_folder, "discussions")
        os.makedirs(discussions_folder, exist_ok=True)
        try:
            discussions = safe_paginate(f"{BASE_API_URL}/courses/{course_id}/discussion_topics?per_page=100")
            for disc in discussions:
                disc_title = make_safe(disc.get("title", "discussion"))
                disc_filename = f"{disc_title}.md"
                disc_path = os.path.join(discussions_folder, disc_filename)
                disc_body = disc.get("message", "")
                disc_author = disc.get("author", {}).get("display_name", "")
                disc_date = disc.get("posted_at", "")
                md = f"# {disc.get('title', 'Discussion') }\n\n*By: {disc_author}*\n*Posted: {disc_date}*\n\n---\n\n{disc_body}"
                with open(disc_path, "w", encoding="utf-8") as f:
                    f.write(md)
                print(f"    ✅ Saved discussion: {disc_filename}")
        except Exception as e:
            print(f"    Error downloading discussions: {e}")

        print("  Downloading files...")
        for file in safe_paginate(
            f"{BASE_API_URL}/courses/{course_id}/files?per_page=100"
        ):
            try:
                file_url = file["url"]
                if file_url in downloaded_file_urls:
                    continue
                r = requests.get(file_url, headers=HEADERS)
                r.raise_for_status()
                file_path = os.path.join(course_folder, make_safe(file["filename"]))
                save_or_unzip(r.content, course_folder, make_safe(file["filename"]))
                downloaded_file_urls.add(file_url)
                print(f"    ✅ Downloaded file: {make_safe(file['filename'])}")
            except Exception as e:
                print(f"    Error downloading {file.get('filename', 'unknown')}: {e}")

        print("  Downloading pages...")
        for page in safe_paginate(
            f"{BASE_API_URL}/courses/{course_id}/pages?per_page=100"
        ):
            try:
                detail = requests.get(
                    f"{BASE_API_URL}/courses/{course_id}/pages/{page['url']}",
                    headers=HEADERS,
                )
                if detail.status_code in [403, 404]:
                    continue
                detail.raise_for_status()
                body = detail.json().get("body", "")
                name = f"page - {page['title']}"
                extract_and_download_linked_files(body, course_folder)
                save_html_as_pdf(course_folder, name, body)
            except Exception as e:
                print(f"    Error handling page {page['title']}: {e}")

        print("  Downloading assignments...")
        for assignment in safe_paginate(
            f"{BASE_API_URL}/courses/{course_id}/assignments?per_page=100"
        ):
            try:
                # Canvas often returns description as null for file-only assignments.
                description_html = assignment.get("description") or ""
                name = f"assignment - {assignment['name']}"
                extract_and_download_linked_files(description_html, course_folder)
                html = f"<h1>{assignment['name']}</h1><p>{description_html}</p>"
                save_html_as_pdf(course_folder, name, html)
            except Exception as e:
                print(f"    Error handling assignment {assignment['name']}: {e}")

        print("  Downloading modules...")
        for module in safe_paginate(
            f"{BASE_API_URL}/courses/{course_id}/modules?per_page=100"
        ):
            try:
                # Start markdown content
                md_content = f"# {module['name']}\n\n"
                items = safe_paginate(
                    f"{BASE_API_URL}/courses/{course_id}/modules/{module['id']}/items?per_page=100"
                )

                for item in items:
                    item_title = item.get("title", "Untitled")
                    item_type = item.get("type", "Unknown")

                    # Build markdown list item with link if available
                    # Prefer File-type items (download via API) to avoid relying on web pages that may require session auth
                    if item.get("type") == "File" and "content_id" in item:
                        # For files, try to get the file URL and download via API
                        try:
                            file_meta = requests.get(
                                f"{BASE_API_URL}/files/{item['content_id']}",
                                headers=HEADERS,
                            )
                            if file_meta.ok:
                                file_data = file_meta.json()
                                file_url = file_data.get("url", "")
                                if file_url:
                                    md_content += (
                                        f"- [{item_title}]({file_url}) ({item_type})\n"
                                    )
                                else:
                                    md_content += f"- {item_title} ({item_type})\n"
                        except Exception:
                            md_content += f"- {item_title} ({item_type})\n"

                        download_canvas_file_by_id(item["content_id"], course_folder)
                    elif "html_url" in item or "url" in item:
                        html_url = item.get("html_url") or item.get("url")
                        md_content += f"- [{item_title}]({html_url}) ({item_type})\n"

                        # If the URL itself points to a Canvas file (e.g., /files/<id>), download it directly via the API
                        m = re.search(r"/files/(\d+)", html_url)
                        if m:
                            file_id = m.group(1)
                            download_canvas_file_by_id(file_id, course_folder)
                        else:
                            # Otherwise fetch the page and parse for linked files (may be behind session auth)
                            item_resp = requests.get(html_url, headers=HEADERS)
                            if item_resp.ok:
                                extract_and_download_linked_files(
                                    item_resp.text, course_folder
                                )
                    elif item.get("type") == "Page" and "page_url" in item:
                        page_url = item["page_url"]
                        page_api_url = (
                            f"{BASE_API_URL}/courses/{course_id}/pages/{page_url}"
                        )
                        # Create a link to the page
                        page_html_url = (
                            f"{CANVAS_DOMAIN}/courses/{course_id}/pages/{page_url}"
                        )
                        md_content += (
                            f"- [{item_title}]({page_html_url}) ({item_type})\n"
                        )

                        page_resp = requests.get(page_api_url, headers=HEADERS)
                        if page_resp.ok:
                            body = page_resp.json().get("body", "")
                            extract_and_download_linked_files(body, course_folder)
                    else:
                        # No link available, just show title and type
                        md_content += f"- {item_title} ({item_type})\n"

                name = f"module - {module['name']}"
                save_markdown(course_folder, name, md_content)
            except Exception as e:
                print(f"    Error saving module {module['name']}: {e}")

        if not DOWNLOAD_SUBMISSIONS:
            print(
                "  ⚠️ Skipping downloading student submissions (DOWNLOAD_SUBMISSIONS=false)"
            )
        else:
            print("  Downloading your submissions...")
            submissions = safe_paginate(
                f"{BASE_API_URL}/courses/{course_id}/students/submissions?per_page=100"
            )
            for sub in submissions:
                for attachment in sub.get("attachments", []):
                    try:
                        file_url = attachment["url"]
                        if file_url in downloaded_file_urls:
                            continue
                        filename = make_safe(f"submission - {attachment['filename']}")
                        r = requests.get(file_url, headers=HEADERS)
                        r.raise_for_status()
                        save_or_unzip(r.content, course_folder, filename)
                        downloaded_file_urls.add(file_url)
                        print(f"    ✅ Downloaded submission: {filename}")
                    except Exception as e:
                        print(f"    Error downloading submission file: {e}")

    print(f"\n✅ All course content downloaded to {DOWNLOADS_BASE}")

    # Optional internal commit/push path. Disabled by default.
    if (
        os.getenv("GITHUB_WORKSPACE")
        and INTERNAL_GIT_COMMIT
        and os.getenv("AUTO_COMMIT", "false").lower() == "true"
    ):
        commit_and_push()


def commit_and_push():
    """Commit and push downloaded files to git (GitHub Actions only)."""
    import subprocess
    from datetime import datetime

    try:
        repo_dir = os.getenv("GITHUB_WORKSPACE")
        auto_push = os.getenv("AUTO_PUSH", "false").lower() == "true"
        target_branch = os.getenv("TARGET_BRANCH", "course")

        # Check if there are any changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            print("ℹ️  No changes to commit.")
            return

        # Add configured output directory relative to repository root.
        output_dir = os.path.basename(os.path.normpath(DOWNLOADS_BASE))
        print("\n📝 Committing downloaded files...")
        subprocess.run(["git", "add", "--", output_dir], cwd=repo_dir, check=True)

        # Create commit message with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Update Canvas course content - {timestamp}"

        subprocess.run(
            ["git", "commit", "-m", commit_message], cwd=repo_dir, check=True
        )
        print(f"✅ Committed changes: {commit_message}")

        # Push to remote only if AUTO_PUSH=true
        if auto_push:
            print(f"🚀 Pushing to remote branch '{target_branch}'...")
            subprocess.run(
                ["git", "push", "origin", f"HEAD:refs/heads/{target_branch}"],
                cwd=repo_dir,
                check=True,
            )
            print("✅ Pushed to remote repository")
        else:
            print(
                "AUTO_PUSH is false; skipping git push. You can manually push to your branch later."
            )

    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error during git operation: {e}")
    except Exception as e:
        print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    main()
