from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import uuid
import json
import requests
import pymupdf
import os
import shutil
import pytesseract
from PIL import Image
import base64
import re
from collections import Counter
import io
import time
from dotenv import load_dotenv

from text_cleaner import process_document

load_dotenv()

# Find Tesseract OCR
TESSERACT_PATH = os.getenv("TESSERACT_PATH") or shutil.which("tesseract")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)

def clean_json_text(text):
    """
    Strips markdown code fences (e.g. ```json ... ```) from Gemini AI responses.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
    return text.strip()


app = Flask(__name__)
CORS(app)

# Maximum upload size: 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DOCUMENT_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "documents"
)
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

MIN_TEXT_LENGTH = 20
OCR_ZOOM = 2.0

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def call_gemini(prompt, image_bytes=None, mime_type=None, temperature=0.2, max_tokens=1500):
    """
    Generic helper to invoke Gemini API with optional multimodal image data.
    Note: REST API expects 'inline_data' and 'mime_type' in snake_case.
    """
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY is not configured.")

    parts = [{"text": prompt}]

    if image_bytes and mime_type:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        parts.insert(0, {
            "inline_data": {
                "mime_type": mime_type,
                "data": b64_data
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        },
        json=payload,
        timeout=60
    )

    if not response.ok:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text
        raise Exception(f"Gemini API error: {error_message}")

    result = response.json()
    try:
        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        return answer.strip()
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected Gemini response structure: {str(e)}")


def ocr_page(page):
    """
    Render a PyMuPDF page as an image and run Tesseract OCR on it.
    """
    matrix = pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM)
    pixmap = page.get_pixmap(matrix=matrix)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    text = pytesseract.image_to_string(image)
    return text.strip()


def ocr_image_file(file_path, extension):
    """
    Extract text from an uploaded image file (PNG, JPG, JPEG) using Tesseract,
    falling back to Gemini Multimodal Vision if Tesseract returns sparse/no text.
    """
    text = ""
    method = "ocr"

    try:
        image = Image.open(file_path).convert('RGB')
        text = pytesseract.image_to_string(image).strip()
    except Exception as e:
        print("Tesseract image error:", str(e))

    if len(text) < MIN_TEXT_LENGTH and GEMINI_API_KEY:
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()

            mime = "image/png" if extension.lower() == "png" else "image/jpeg"
            vision_prompt = (
                "You are an expert OCR system. Extract all readable text from this image accurately. "
                "Preserve titles, numbers, bullet points, and tables. Return ONLY the raw extracted text."
            )
            vision_text = call_gemini(vision_prompt, image_bytes=img_bytes, mime_type=mime)
            if vision_text and len(vision_text) > len(text):
                return vision_text, "gemini_vision"
        except Exception as ve:
            print("Gemini Vision OCR fallback error:", str(ve))

    return text, method


def extract_page_text(page):
    """
    Extract text from a single PDF page, preferring native PyMuPDF extraction,
    falling back to Tesseract OCR, and then Gemini Vision OCR if needed.
    """
    text = page.get_text("text").strip()

    if len(text) >= MIN_TEXT_LENGTH:
        return text, "text"

    # Fallback 1: Tesseract OCR
    ocr_text = ""
    try:
        ocr_text = ocr_page(page)
        if len(ocr_text) >= MIN_TEXT_LENGTH:
            return ocr_text, "ocr"
    except Exception as e:
        print("OCR error on page:", str(e))

    # Fallback 2: Gemini Vision OCR
    if GEMINI_API_KEY:
        try:
            matrix = pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM)
            pixmap = page.get_pixmap(matrix=matrix)
            img_bytes = pixmap.tobytes("png")
            vision_prompt = (
                "Extract all text from this page image accurately. Return ONLY the extracted text."
            )
            v_text = call_gemini(vision_prompt, image_bytes=img_bytes, mime_type="image/png")
            if v_text:
                return v_text, "gemini_vision"
        except Exception as ve:
            print("Gemini Vision PDF fallback error:", str(ve))

    return ocr_text or text, "ocr" if ocr_text else "text"



def find_relevant_pages(document, query):
    """
    Advanced multi-keyword and phrase ranking algorithm to retrieve top relevant pages.
    """
    query_lower = query.lower()
    clean_query = re.sub(r'[^\w\s]', '', query_lower)
    query_words = [w for w in clean_query.split() if len(w) > 2 and w not in STOP_WORDS]

    whole_document_keywords = [
        "whole pdf", "whole document", "entire pdf", "entire document", "all pages",
        "every page", "full pdf", "full document", "document summary", "summarize",
        "what is written", "contain", "overview", "executive summary"
    ]

    pages = document.get("content", [])
    if not pages:
        return []

    # Check if query requests full document context
    is_full_doc_request = any(kw in query_lower for kw in whole_document_keywords)

    if is_full_doc_request:
        results = []
        for index, page in enumerate(pages, start=1):
            text = page.get("text", "")
            if not text:
                continue
            page_number = page.get("page_number") or page.get("page") or index
            results.append({
                "page": page_number,
                "text": text,
                "score": 10
            })
        return results[:10]

    results = []
    for index, page in enumerate(pages, start=1):
        text = page.get("text", "")
        if not text:
            continue

        text_lower = text.lower()
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        score = 0

        # Exact phrase bonus
        if clean_query and clean_query in clean_text:
            score += 15

        # Individual word matches
        for word in query_words:
            count = clean_text.split().count(word)
            if count > 0:
                score += min(count * 2, 8)

        # Partial word matches
        for word in query_words:
            if word in text_lower and not clean_text.split().count(word):
                score += 1

        if score > 0:
            page_number = page.get("page_number") or page.get("page") or index
            results.append({
                "page": page_number,
                "text": text,
                "score": score
            })

    # Sort highest relevance first
    results.sort(key=lambda x: x["score"], reverse=True)

    # Return top 5 relevant pages
    return results[:5]


def generate_ai_answer(question, page_results):
    """
    Generates a RAG answer grounding Gemini in context from top relevant pages.
    """
    if not page_results:
        return "I couldn't find relevant sections in the document to answer this question.", []

    context_blocks = []
    cited_pages = []
    for item in page_results:
        p_num = item.get("page")
        p_text = item.get("text", "")
        cited_pages.append(p_num)
        context_blocks.append(f"=== Page {p_num} ===\n{p_text}")

    source_context = "\n\n".join(context_blocks)

    prompt = f"""
You are DocuMind, an elite AI Document Intelligence Assistant.

Answer the user's question accurately, thoroughly, and strictly using the document excerpts provided below.

INSTRUCTIONS:
1. Ground every statement in the provided document text.
2. Directly answer the question in clear, polished formatting (use markdown formatting like bold text, lists, or tables if helpful).
3. Explicitly cite page numbers in your answer whenever stating key facts (e.g. "[Page 12]" or "[Page 14]").
4. If the exact answer is not present in the excerpts, clearly explain what relevant information is available or state that the specific details are not mentioned in the document.

User Question:
{question}

Document Excerpts:
{source_context}
"""

    answer = call_gemini(prompt, temperature=0.2, max_tokens=1200)
    return answer, cited_pages


# ---------------------------------------
# HEALTH CHECK
# ---------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "DocuMind backend is running",
        "gemini_configured": bool(GEMINI_API_KEY),
        "tesseract_configured": bool(TESSERACT_PATH and os.path.exists(TESSERACT_PATH))
    })


# ---------------------------------------
# DOCUMENT UPLOAD (PDF & Image OCR)
# ---------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type. Use PDF, PNG, JPG, or JPEG."}), 400

    document_id = str(uuid.uuid4())
    extension = file.filename.rsplit(".", 1)[1].lower()
    safe_filename = f"{document_id}.{extension}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

    file.save(file_path)
    file_size = os.path.getsize(file_path)

    page_count = 1
    pages = []

    try:
        if extension == "pdf":
            doc = pymupdf.open(file_path)
            page_count = len(doc)

            for page_number, page in enumerate(doc, start=1):
                text, extraction_method = extract_page_text(page)
                pages.append({
                    "page": page_number,
                    "text": text,
                    "extraction_method": extraction_method
                })
            doc.close()

        else:
            # Image OCR (PNG / JPG / JPEG)
            img_text, extraction_method = ocr_image_file(file_path, extension)
            pages.append({
                "page": 1,
                "text": img_text,
                "extraction_method": extraction_method
            })
            page_count = 1

        cleaned_pages, combined_clean_text = process_document(pages)

        document_data = {
            "document_id": document_id,
            "filename": file.filename,
            "extension": extension,
            "size": file_size,
            "pages": page_count,
            "upload_timestamp": time.time(),
            "content": pages,
            "cleaned_content": cleaned_pages,
            "combined_clean_text": combined_clean_text
        }

        text_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
        with open(text_file, "w", encoding="utf-8") as f:
            json.dump(document_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "size": file_size,
            "pages": page_count,
            "pages_extracted": len(pages),
            "message": "Document uploaded and processed successfully"
        })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            "success": False,
            "error": f"Could not process document: {str(e)}"
        }), 400


# ---------------------------------------
# DOCUMENT SEARCH
# ---------------------------------------

@app.route("/api/search", methods=["POST"])
def search_document():
    data = request.get_json() or {}
    document_id = data.get("document_id")
    query = data.get("query", "").strip()

    if not document_id or not query:
        return jsonify({"success": False, "error": "document_id and query are required"}), 400

    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        results = find_relevant_pages(document, query)
        return jsonify({
            "success": True,
            "query": query,
            "results": results[:5]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# AI DOCUMENT Q&A (RAG with Citations)
# ---------------------------------------

@app.route("/api/qa", methods=["POST"])
def document_qa():
    data = request.get_json() or {}
    document_id = data.get("document_id")
    question = data.get("question", "").strip()

    if not document_id or not question:
        return jsonify({"success": False, "error": "document_id and question are required"}), 400

    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        results = find_relevant_pages(document, question)

        if not results:
            # Fallback to first few pages if no keyword match
            content = document.get("content", [])
            for idx, p in enumerate(content[:3], start=1):
                results.append({
                    "page": p.get("page") or idx,
                    "text": p.get("text", ""),
                    "score": 1
                })

        answer, cited_pages = generate_ai_answer(question, results)

        primary_page = cited_pages[0] if cited_pages else None
        source_snippets = [
            {"page": r["page"], "text": r["text"]}
            for r in results
        ]

        combined_source = "\n\n".join([f"Page {r['page']}:\n{r['text']}" for r in results])

        return jsonify({
            "success": True,
            "answer": answer,
            "page": primary_page,
            "pages": cited_pages,
            "source_text": combined_source,
            "sources": source_snippets
        })

    except Exception as e:
        print("Q&A Error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# AI DOCUMENT SUMMARIZATION
# ---------------------------------------

@app.route("/api/summarize", methods=["POST"])
def summarize_document():
    data = request.get_json() or {}
    document_id = data.get("document_id")
    summary_type = data.get("summary_type", "executive")

    if not document_id:
        return jsonify({"success": False, "error": "document_id is required"}), 400

    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        text = document.get("combined_clean_text") or ""
        if not text:
            _, text = process_document(document.get("content", []))

        filename = document.get("filename", "Document")

        max_chars = 40000
        truncated_text = text[:max_chars]

        prompt = f"""
You are an expert Document Intelligence Analyst.
Analyze the following uploaded document ("{filename}") and provide a comprehensive, beautifully structured JSON summary.

DOCUMENT CONTENT:
{truncated_text}

OUTPUT INSTRUCTIONS:
Return a strictly valid JSON object matching this structure (no extra markdown outside the JSON block):
{{
    "executive_summary": "High-level 2-3 paragraph summary of the document purpose, scope, and key points.",
    "key_takeaways": [
        "Takeaway 1",
        "Takeaway 2",
        "Takeaway 3",
        "Takeaway 4",
        "Takeaway 5"
    ],
    "key_metrics_and_numbers": [
        "Metric/Requirement 1 (e.g., Minimum attendance requirement: 75%)",
        "Metric/Requirement 2 (e.g., Application deadline: Oct 15)",
        "Metric/Requirement 3",
        "Metric/Requirement 4"
    ],
    "action_items_and_compliance": [
        "Important rule/action item 1",
        "Important rule/action item 2",
        "Important rule/action item 3"
    ]
}}
"""

        summary_json_str = call_gemini(prompt, temperature=0.2, max_tokens=1500)
        clean_json = clean_json_text(summary_json_str)

        try:
            summary_data = json.loads(clean_json)
        except Exception:
            summary_data = {
                "executive_summary": summary_json_str,
                "key_takeaways": ["Analysis completed."],
                "key_metrics_and_numbers": ["Extracted from document."],
                "action_items_and_compliance": ["Review source document."]
            }

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "summary": summary_data
        })

    except Exception as e:
        print("Summarization error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# KEYWORD & ENTITY EXTRACTION
# ---------------------------------------

@app.route("/api/keywords", methods=["POST"])
def extract_keywords():
    data = request.get_json() or {}
    document_id = data.get("document_id")

    if not document_id:
        return jsonify({"success": False, "error": "document_id is required"}), 400

    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        text = document.get("combined_clean_text") or ""

        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in STOP_WORDS]
        freq_counter = Counter(words)
        top_freq = [{"keyword": k, "count": v} for k, v in freq_counter.most_common(12)]

        truncated_text = text[:30000]
        prompt = f"""
Analyze the document text below and extract structured metadata, key topics, named entities, and regulations.

DOCUMENT TEXT:
{truncated_text}

OUTPUT INSTRUCTIONS:
Return a strictly valid JSON object with the following schema:
{{
    "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4"],
    "entities": ["Organization 1", "Standard/Policy Name", "Location/Department"],
    "requirements_and_numbers": ["Requirement 1", "Rule/Percentage 2"],
    "top_keywords": ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4", "Keyword 5", "Keyword 6", "Keyword 7", "Keyword 8"]
}}
"""
        ai_json_str = call_gemini(prompt, temperature=0.1, max_tokens=1000)
        clean_json = clean_json_text(ai_json_str)

        try:
            extracted = json.loads(clean_json)
        except Exception:
            extracted = {
                "topics": [item["keyword"].title() for item in top_freq[:4]],
                "entities": [],
                "requirements_and_numbers": [],
                "top_keywords": [item["keyword"] for item in top_freq[:8]]
            }

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": document.get("filename"),
            "frequency_keywords": top_freq,
            "extracted": extracted
        })

    except Exception as e:
        print("Keyword Extraction Error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# DOCUMENT COMPARISON (Doc A vs Doc B)
# ---------------------------------------

@app.route("/api/compare", methods=["POST"])
def compare_documents():
    data = request.get_json() or {}
    document_id_1 = data.get("document_id_1")
    document_id_2 = data.get("document_id_2")

    if not document_id_1 or not document_id_2:
        return jsonify({"success": False, "error": "Both document_id_1 and document_id_2 are required"}), 400

    file1 = os.path.join(DOCUMENT_FOLDER, f"{document_id_1}.json")
    file2 = os.path.join(DOCUMENT_FOLDER, f"{document_id_2}.json")

    if not os.path.exists(file1) or not os.path.exists(file2):
        return jsonify({"success": False, "error": "One or both documents were not found"}), 404

    try:
        with open(file1, "r", encoding="utf-8") as f:
            doc1 = json.load(f)
        with open(file2, "r", encoding="utf-8") as f:
            doc2 = json.load(f)

        text1 = doc1.get("combined_clean_text", "")[:20000]
        text2 = doc2.get("combined_clean_text", "")[:20000]

        prompt = f"""
You are a senior Document Intelligence Analyst.
Compare Document 1 ("{doc1.get('filename')}") and Document 2 ("{doc2.get('filename')}") side-by-side.

DOCUMENT 1 ("{doc1.get('filename')}"):
{text1}

DOCUMENT 2 ("{doc2.get('filename')}"):
{text2}

OUTPUT INSTRUCTIONS:
Return a strictly valid JSON object matching this schema:
{{
    "overview_comparison": "Comprehensive overview comparing the scope, domain, and purpose of both documents.",
    "key_similarities": [
        "Similarity point 1",
        "Similarity point 2",
        "Similarity point 3"
    ],
    "key_differences": [
        "Difference point 1",
        "Difference point 2",
        "Difference point 3",
        "Difference point 4"
    ],
    "policy_updates_or_changes": [
        "Notable policy change or quantitative difference between Doc 1 and Doc 2"
    ],
    "summary_recommendation": "Final recommendation or conclusion based on the comparison."
}}
"""

        comp_json_str = call_gemini(prompt, temperature=0.2, max_tokens=1500)
        clean_json = clean_json_text(comp_json_str)

        try:
            comparison_data = json.loads(clean_json)
        except Exception:
            comparison_data = {
                "overview_comparison": comp_json_str,
                "key_similarities": [],
                "key_differences": [],
                "policy_updates_or_changes": [],
                "summary_recommendation": ""
            }

        return jsonify({
            "success": True,
            "doc1": {"id": document_id_1, "filename": doc1.get("filename")},
            "doc2": {"id": document_id_2, "filename": doc2.get("filename")},
            "comparison": comparison_data
        })

    except Exception as e:
        print("Comparison error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# DOWNLOADABLE INTELLIGENCE REPORT
# ---------------------------------------

@app.route("/api/report/<document_id>", methods=["GET"])
def export_report(document_id):
    fmt = request.args.get("format", "md").lower()
    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")

    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            doc = json.load(f)

        filename = doc.get("filename", "Document")
        pages_count = doc.get("pages", 1)
        text = doc.get("combined_clean_text", "")

        report_md = f"""# DocuMind AI Intelligence Report

**Document Name:** {filename}  
**Document ID:** {document_id}  
**Total Pages:** {pages_count}  
**Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary

This document intelligence report was automatically compiled by **DocuMind AI**. Below is the extracted content overview, page index, and structured textual analysis.

### Content Overview
- **Extracted Pages:** {pages_count} page(s)
- **Extraction Methods Used:** Native Text Extraction & Tesseract OCR
- **Character Count:** {len(text)} characters

---

## 2. Extracted Document Content

{text}

---

*Report created by DocuMind AI Document Intelligence System*
"""

        if fmt == "html":
            html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DocuMind Report - {filename}</title>
<style>
body {{ font-family: system-ui, sans-serif; line-height: 1.6; max-width: 850px; margin: 40px auto; padding: 0 20px; color: #111; }}
h1, h2, h3 {{ color: #046C4E; }}
pre {{ background: #f4f7f5; padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
hr {{ border: none; border-top: 1px solid #dce6e1; margin: 30px 0; }}
</style>
</head>
<body>
<h1>DocuMind AI Intelligence Report</h1>
<p><strong>Document:</strong> {filename} | <strong>Pages:</strong> {pages_count} | <strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<hr>
<h2>Extracted Text Content</h2>
<pre>{text}</pre>
</body>
</html>"""
            return Response(
                html_content,
                mimetype="text/html",
                headers={"Content-Disposition": f"attachment;filename=DocuMind_Report_{document_id}.html"}
            )

        elif fmt == "txt":
            plain_txt = f"DocuMind Intelligence Report\nDocument: {filename}\nPages: {pages_count}\n\n" + text
            return Response(
                plain_txt,
                mimetype="text/plain",
                headers={"Content-Disposition": f"attachment;filename=DocuMind_Report_{document_id}.txt"}
            )

        else:
            return Response(
                report_md,
                mimetype="text/markdown",
                headers={"Content-Disposition": f"attachment;filename=DocuMind_Report_{document_id}.md"}
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# DOCUMENT MANAGEMENT & LISTING
# ---------------------------------------

@app.route("/api/documents", methods=["GET"])
def list_documents():
    docs = []
    try:
        for fname in os.listdir(DOCUMENT_FOLDER):
            if fname.endswith(".json"):
                fpath = os.path.join(DOCUMENT_FOLDER, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        docs.append({
                            "document_id": data.get("document_id"),
                            "filename": data.get("filename"),
                            "pages": data.get("pages"),
                            "size": data.get("size"),
                            "upload_timestamp": data.get("upload_timestamp", 0)
                        })
                except Exception:
                    continue
        
        docs.sort(key=lambda x: x.get("upload_timestamp", 0), reverse=True)
        return jsonify({"success": True, "documents": docs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/document/<document_id>", methods=["GET"])
def get_document(document_id):
    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return jsonify({"success": True, "document": doc})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/clean/<document_id>", methods=["GET"])
def get_cleaned_document(document_id):
    document_file = os.path.join(DOCUMENT_FOLDER, f"{document_id}.json")
    if not os.path.exists(document_file):
        return jsonify({"success": False, "error": "Document not found"}), 404

    try:
        with open(document_file, "r", encoding="utf-8") as f:
            document = json.load(f)

        if "cleaned_content" not in document or "combined_clean_text" not in document:
            cleaned_pages, combined_clean_text = process_document(document.get("content", []))
            document["cleaned_content"] = cleaned_pages
            document["combined_clean_text"] = combined_clean_text
            with open(document_file, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": document.get("filename"),
            "pages": document.get("pages"),
            "cleaned_content": document.get("cleaned_content", []),
            "combined_clean_text": document.get("combined_clean_text", "")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------
# START SERVER
# ---------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000
    )