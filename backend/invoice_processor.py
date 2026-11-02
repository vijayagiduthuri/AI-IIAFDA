import io
import re
from pathlib import Path

import pandas as pd
import pymupdf
import pytesseract

from PIL import Image, ImageOps, ImageEnhance

from backend.config import INVOICES_FILE, IMAGES_DIR


# --------------------------------------------------
# SUPPORTED FILE FORMATS
# --------------------------------------------------

SUPPORTED_IMAGE_FORMATS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}

SUPPORTED_PDF_FORMATS = {".pdf"}


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    """Prepare an invoice image for better OCR readability."""

    image = ImageOps.exif_transpose(image)
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


# --------------------------------------------------
# IMAGE OCR
# --------------------------------------------------

def extract_text_from_image(image):
    """Extract text from a PIL image using Tesseract."""

    processed_image = preprocess_image(image)

    return pytesseract.image_to_string(
        processed_image,
        config="--oem 3 --psm 6"
    ).strip()


# --------------------------------------------------
# PDF TEXT EXTRACTION AND OCR
# --------------------------------------------------

def extract_text_from_pdf(pdf_bytes):
    """Extract selectable text or OCR scanned PDF pages."""

    extracted_pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text").strip()

            # Use OCR for scanned pages
            if not page_text:

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )

                page_text = extract_text_from_image(image)

            extracted_pages.append(
                f"--- Page {page_number} ---\n{page_text}"
            )

    return "\n\n".join(extracted_pages).strip()


# --------------------------------------------------
# MAIN INVOICE TEXT EXTRACTION
# --------------------------------------------------

def extract_invoice_text(file_path=None, uploaded_file=None):
    """Extract text from an invoice image or PDF."""

    if uploaded_file is not None:

        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

    elif file_path is not None:

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Invoice not found: {path}"
            )

        filename = path.name
        file_bytes = path.read_bytes()

    else:
        raise ValueError(
            "Provide file_path or uploaded_file."
        )

    extension = Path(filename).suffix.lower()

    if extension in SUPPORTED_IMAGE_FORMATS:

        with Image.open(io.BytesIO(file_bytes)) as image:
            extracted_text = extract_text_from_image(image)

    elif extension in SUPPORTED_PDF_FORMATS:

        extracted_text = extract_text_from_pdf(file_bytes)

    else:
        raise ValueError(
            f"Unsupported invoice format: {extension}"
        )

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "status": "success" if extracted_text else "empty_text",
        "message": (
            "Invoice text extracted successfully."
            if extracted_text
            else "No readable text found."
        )
    }


# --------------------------------------------------
# INVOICE FIELD EXTRACTION
# --------------------------------------------------

def _clean_supplier_id(raw_value):
    """
    Repair a UUID read by OCR, e.g.
      'cdbc2c1c-a4af -47bd-ae45-5cc6b88e830f' -> 'cdbc2c1c-a4af-47bd-ae45-5cc6b88e830f'
    Common OCR letter/digit confusions (O->0, l/I->1, S->5) are fixed.
    Returns None if the value cannot be turned into a valid UUID.
    """

    text = "".join(str(raw_value).split()).replace("-", "").lower()

    ocr_fixes = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5"})
    text = text.translate(ocr_fixes)

    if len(text) != 32 or any(c not in "0123456789abcdef" for c in text):
        return None

    return (
        f"{text[0:8]}-{text[8:12]}-{text[12:16]}-"
        f"{text[16:20]}-{text[20:32]}"
    )


def extract_invoice_fields(ocr_text):
    """Extract basic invoice fields from OCR text."""

    patterns = {
        # OCR sometimes inserts a space, e.g. "INV_ 9000001";
        # spaces are removed afterwards.
        "invoice_id": (
            r"(?im)^\s*Invoice\s*ID\s*:\s*([A-Za-z0-9_-][A-Za-z0-9_ \t-]*?)\s*$"
        ),
        # Supplier IDs are UUIDs. Tesseract often inserts spaces or
        # breaks them, so allow spaces/dashes anywhere and clean up
        # afterwards in _clean_supplier_id().
        "supplier_id": (
            r"(?im)^\s*Supplier\s*ID\s*[:;.]?\s*([0-9A-Za-z][0-9A-Za-z \t-]{20,60}?)\s*$"
        ),
        # Supplier country code printed under the supplier ID
        "supplier_country": (
            r"(?im)^\s*Country\s*[:;.]?\s*([A-Za-z]{2})\s*$"
        ),
        "invoice_date": (
            r"(?im)^\s*Invoice\s*Date\s*:\s*(\d{4}-\d{2}-\d{2})\s*$"
        ),
        "payment_terms": (
            r"(?im)^\s*Payment\s*Terms\s*:\s*([A-Za-z0-9]+)\s*$"
        ),
        "subtotal": (
            r"(?im)^\s*Subtotal\s*:\s*([\d,]+(?:\.\d{1,2})?)\s*$"
        ),
        "tax": (
            r"(?im)^\s*Tax\s*\([^)]*\)\s*:\s*([\d,]+(?:\.\d{1,2})?)\s*$"
        ),
        "total": (
            r"(?im)^\s*Total\s*:\s*([\d,]+(?:\.\d{1,2})?)"
            r"\s*([A-Z]{3})?\s*$"
        )
    }

    extracted_fields = {}

    for field, pattern in patterns.items():

        match = re.search(pattern, ocr_text)

        if match:

            value = match.group(1).replace(",", "")

            if field in {"subtotal", "tax", "total"}:
                extracted_fields[field] = float(value)

            elif field == "supplier_id":
                extracted_fields[field] = _clean_supplier_id(
                    match.group(1)
                )

            elif field == "invoice_id":
                extracted_fields[field] = re.sub(r"\s+", "", value)

            elif field in {"supplier_country", "payment_terms"}:
                extracted_fields[field] = value.upper()

            else:
                extracted_fields[field] = value

            if field == "total":
                extracted_fields["currency"] = match.group(2)

        else:
            extracted_fields[field] = None

    if "currency" not in extracted_fields:
        extracted_fields["currency"] = None

    return extracted_fields


# --------------------------------------------------
# REAL DATASET OCR AND FIELD EXTRACTION TEST
# --------------------------------------------------

def test_dataset_invoice_ocr():

    invoices = pd.read_parquet(INVOICES_FILE)

    sample_rows = invoices.loc[
        invoices["image_path"].notna()
    ]

    if sample_rows.empty:
        print("No invoice image paths found.")
        return

    sample_invoice = sample_rows.iloc[0]

    image_filename = Path(
        str(sample_invoice["image_path"])
    ).name

    image_path = IMAGES_DIR / image_filename

    print("=" * 60)
    print("INVOICE OCR AND FIELD EXTRACTION TEST")
    print("=" * 60)

    print("Invoice ID:", sample_invoice["invoice_id"])
    print("Image:", image_path)

    if not image_path.is_file():
        print("Image file not found!")
        return

    result = extract_invoice_text(
        file_path=image_path
    )

    print("\nOCR Status:", result["status"])

    if result["status"] != "success":
        print(result["message"])
        return

    print("\nEXTRACTED OCR TEXT")
    print("-" * 60)
    print(result["extracted_text"])

    fields = extract_invoice_fields(
        result["extracted_text"]
    )

    print("\nSTRUCTURED INVOICE FIELDS")
    print("-" * 60)

    for field, value in fields.items():
        print(f"{field}: {value}")


# --------------------------------------------------
# EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("Invoice OCR processor initialized.")

    test_dataset_invoice_ocr()