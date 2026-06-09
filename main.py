"""
ProductsHandler
---------------
Triggered by EvolutionCleaner after product images are staged/approved.
Receives a list of approved media_urls from product_image_staging.

Responsibilities:
  1. Exact binary deduplication via MD5 hash grouping.
  2. Thumbnail AI filter check (gpt-4o-mini, low-detail, no multiplier unless tokens are too high).
  3. Semantic deduplication check against the current batch items to group alternative views.
  4. Download high-quality source images and run deep description analysis (always multiplied cost).
  5. Save to products table with status='discovered'.
  6. Log AI + storage usage for billing reconciliation.
"""

import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ============================================================
# SECTION 1 — CONFIG
# ============================================================

openai_key   = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")
STORAGE_BUCKET    = os.getenv("PRODUCT_IMAGE_BUCKET", "product-images")

client_ai : AsyncOpenAI = AsyncOpenAI(api_key=openai_key)
supabase  : Client       = create_client(supabase_url, supabase_key)

VISION_MODEL    = "gpt-4o-mini"
MAX_IMAGES_PER_RUN = int(os.getenv("MAX_IMAGES_PER_RUN", "50"))

VISION_INPUT_COST_PER_TOKEN  = 0.000000150
VISION_OUTPUT_COST_PER_TOKEN = 0.000000600
STORAGE_COST_PER_MB          = 0.000021

# ============================================================
# SECTION 2 — USAGE LOGGING
# ============================================================

def log_ai_usage(
    business_id:       str,
    run_id:            str,
    bot_id:            str,
    model:             str,
    prompt_tokens:     int,
    completion_tokens: int,
    multiplier:        float = 1.0
) -> float:
    try:
        total_tokens = prompt_tokens + completion_tokens
        
        # Rule: No multiplier for the first two runs UNLESS tokens are too high
        if bot_id in ["thumbnail_filter", "semantic_dedup"] and total_tokens > 1500:
            multiplier = 5.0

        cost = round(
            (prompt_tokens     * VISION_INPUT_COST_PER_TOKEN +
             completion_tokens * VISION_OUTPUT_COST_PER_TOKEN) * multiplier,
            6
        )
        supabase.table("ai_usage_log").insert({
            "business_id":        business_id,
            "run_id":             run_id,
            "bot_id":             bot_id,
            "model":              model,
            "input_type":         "vision",
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "total_tokens":       total_tokens,
            "estimated_cost_usd": cost,
            "created_at":         datetime.now(timezone.utc).isoformat()
        }).execute()
        return cost
    except Exception as e:
        print(f"  [Usage] AI log failed: {e}")
        return 0.0


def log_storage_usage(
    business_id:     str,
    run_id:          str,
    file_path:       str,
    file_size_bytes: int
):
    try:
        supabase.table("storage_usage_log").insert({
            "business_id":     business_id,
            "run_id":          run_id,
            "bucket":          STORAGE_BUCKET,
            "file_path":       file_path,
            "file_size_bytes": file_size_bytes,
            "created_at":      datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        print(f"  [Usage] Storage log failed: {e}")

# ============================================================
# SECTION 3 — IMAGE DOWNLOAD + STORAGE UPLOAD
# ============================================================

async def download_image(url: str) -> bytes | None:
    """Download from Evolution CDN or any direct URL."""
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            headers = {"apikey": EVOLUTION_API_KEY} if EVOLUTION_API_KEY else {}
            r = await http.get(url, headers=headers, follow_redirects=True)
            if r.status_code == 200:
                return r.content
            print(f"  [Download] HTTP {r.status_code} for {url[:60]}")
    except Exception as e:
        print(f"  [Download] Failed: {e}")
    return None


def upload_to_storage(
    business_id: str,
    run_id:      str,
    image_bytes: bytes,
    index:       int
) -> tuple[str | None, str]:
    """
    Upload to Supabase Storage.
    Returns (public_url, file_path). public_url is None on failure.
    """
    file_path = f"{business_id}/discovered/{run_id}_{index}.jpg"
    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path         = file_path,
            file         = image_bytes,
            file_options = {"content-type": "image/jpeg", "upsert": "true"}
        )
        public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(file_path)
        return public_url, file_path
    except Exception as e:
        print(f"  [Storage] Upload failed: {e}")
        return None, file_path

# ============================================================
# SECTION 4 — VISION PIPELINE RUNS
# ============================================================

async def filter_thumbnail_ai(image_url: str, business_info: dict, business_id: str, run_id: str) -> dict | None:
    """RUN 1: Check if the thumbnail is a valid product based on the profile context."""
    business_name = business_info.get("name", "Unknown")
    biz_type      = business_info.get("business_type", "general")
    
    system_prompt = f"""You are a product screening helper for "{business_name}" ({biz_type}).
Analyze this low-resolution image thumbnail and determine if it represents an items/service catalog offering.
Filter out junk, payment/M-Pesa screenshots, memes, text receipts, personal chat selfies, or bad images.

Return ONLY a valid JSON object:
{{
  "is_product": true or false,
  "reason": "A 5-word summary description of what this item is"
}}"""

    try:
        response = await client_ai.chat.completions.create(
            model           = VISION_MODEL,
            max_tokens      = 150,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                        {"type": "text",      "text": "Is this a valid catalog item?"}
                    ]
                }
            ]
        )
        usage = response.usage
        log_ai_usage(
            business_id       = business_id,
            run_id            = run_id,
            bot_id            = "thumbnail_filter",
            model             = VISION_MODEL,
            prompt_tokens     = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            multiplier        = 1.0
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  [Filter AI] Failed: {e}")
        return None


async def check_semantic_duplicate(image_url: str, item_reason: str, current_batch: list[dict], business_id: str, run_id: str) -> dict | None:
    """RUN 2: Check if this item is a semantic duplicate (different angle/duplicate) of another item processed in this batch."""
    batch_items_str = json.dumps([{"title": i["title"], "description": i["description"]} for i in current_batch])
    
    system_prompt = f"""You are checking for catalog item duplicates.
We have already identified these products in the current batch processing run:
{batch_items_str}

Analyze this new image thumbnail (identified as: {item_reason}). Determine if it shows the EXACT SAME product/item model (e.g. an alternate photo angle or a re-uploaded match) as one of the items listed above.

Return ONLY a valid JSON object:
{{
  "is_duplicate": true or false,
  "matched_title": "Title of the item it matches, or null if false"
}}"""

    try:
        response = await client_ai.chat.completions.create(
            model           = VISION_MODEL,
            max_tokens      = 150,
            temperature     = 0.1,
            response_format = {"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                        {"type": "text",      "text": "Is this an alternate view or duplicate item?"}
                    ]
                }
            ]
        )
        usage = response.usage
        log_ai_usage(
            business_id       = business_id,
            run_id            = run_id,
            bot_id            = "semantic_dedup",
            model             = VISION_MODEL,
            prompt_tokens     = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            multiplier        = 1.0
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  [Semantic Dedup] Failed: {e}")
        return None


async def analyse_image(
    image_url:     str,
    business_info: dict,
    business_id:   str,
    run_id:        str
) -> dict | None:
    """RUN 3: Call GPT-4o-mini vision to extract structured product metadata. ALWAYS multiplied cost."""
    business_name = business_info.get("name", "Unknown")
    currency      = business_info.get("currency", "KES")
    biz_type      = business_info.get("business_type", "general")

    system_prompt = f"""You are a product cataloguing assistant for an African SME called "{business_name}".
Business type: {biz_type}. Currency: {currency}.

Analyse the product image and return ONLY a valid JSON object. No preamble. No markdown fences.

Return:
{{
  "title":             "Short clear product name (max 6 words)",
  "description_short": "One sentence (max 20 words)",
  "description_long":  "2-3 sentences. Mention materials, use case, or variants if visible.",
  "product_type":      "Category e.g. clothing, electronics, food, furniture, beauty",
  "style":             "Style descriptor if applicable. null if not applicable.",
  "materials":         ["list", "of", "visible", "materials"],
  "key_features":      ["up to 5 short feature tags"],
  "patterns":          ["visible patterns e.g. striped, floral, plain. Empty if none."],
  "occasion":          ["suitable occasions. Empty if not applicable."],
  "color_palette":     ["dominant colours"],
  "price_visible":     true or false,
  "price_estimate":    numeric price in {currency} if visible or inferable. null if unknown.,
  "confidence":        "high | medium | low"
}}"""

    try:
        response = await client_ai.chat.completions.create(
            model           = VISION_MODEL,
            max_tokens      = 600,
            temperature     = 0.2,
            response_format = {"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                        {"type": "text",      "text": "Analyse this product image and return the JSON."}
                    ]
                }
            ]
        )
        usage = response.usage
        log_ai_usage(
            business_id       = business_id,
            run_id            = run_id,
            bot_id            = "products_handler",
            model             = VISION_MODEL,
            prompt_tokens     = usage.prompt_tokens,
            completion_tokens = usage.completion_tokens,
            multiplier        = 5.0 # Always multipling cost on the final run
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  [Vision] Analysis failed: {e}")
        return None

# ============================================================
# SECTION 5 — DEDUP + INSERT
# ============================================================

def get_existing_handles(business_id: str) -> set:
    """Load all handles already in products table for this business."""
    try:
        res = supabase.table("products") \
            .select("handle") \
            .eq("business_id", business_id) \
            .execute()
        return {r["handle"] for r in (res.data or []) if r.get("handle")}
    except Exception:
        return set()


def build_handle(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', title.lower().strip()).strip('-')


def insert_product(business_id: str, ai_result: dict, stored_url: str, is_ecommerce: bool) -> str | None:
    """Build and insert a product row with status='discovered'."""
    title  = ai_result.get("title") or "Unnamed Product"
    handle = build_handle(title)

    price_val = ai_result.get("price_estimate")
    if price_val is not None:
        try:
            price_val = float(price_val)
        except (TypeError, ValueError):
            price_val = None

    payload = {
        "id":                str(uuid.uuid4()),
        "business_id":       business_id,
        "title":             title,
        "description_short": ai_result.get("description_short") or "",
        "description_long":  ai_result.get("description_long")  or "",
        "price":             price_val,
        "images":            [stored_url],
        "type":              "product" if is_ecommerce else "service",
        "status":            "discovered", # Appears as discovered on frontend
        "source":            "discovered",
        "handle":            handle,
        "product_type":      ai_result.get("product_type"),
        "style":             ai_result.get("style"),
        "materials":         ai_result.get("materials")     or [],
        "key_features":      ai_result.get("key_features")  or [],
        "patterns":          ai_result.get("patterns")      or [],
        "occasion":          ai_result.get("occasion")      or [],
        "color_palette":     ai_result.get("color_palette") or [],
        "is_visible":        True,
        "needs_polish":      True,
        "stock_quantity":    0,
        "vectorized":        False,
    }

    try:
        supabase.table("products").insert(payload).execute()
        return handle
    except Exception as e:
        print(f"  [Insert] Failed for '{title}': {e}")
        return None

# ============================================================
# SECTION 6 — MAIN PIPELINE
# ============================================================

async def run_products_pipeline(business_id: str, approved_media_urls: list[str]):
    """Full architectural pipeline for processing discovery images sequentially."""
    run_id = str(uuid.uuid4())
    print(f"\n[ProductsHandler] Starting — business:{business_id} run:{run_id}")

    try:
        db_res = supabase.table("product_image_staging") \
            .select("media_url") \
            .eq("business_id", business_id) \
            .eq("status", "approved") \
            .execute()
        db_approved_urls = {r["media_url"] for r in (db_res.data or []) if r.get("media_url")}
    except Exception as e:
        print(f"  [ProductsHandler] DB read failed, using trigger list only: {e}")
        db_approved_urls = set()

    all_urls = list(dict.fromkeys(approved_media_urls + list(db_approved_urls)))
    all_urls = all_urls[:MAX_IMAGES_PER_RUN]

    if not all_urls:
        print(f"  [ProductsHandler] No approved images — exiting")
        return

    print(f"  [ProductsHandler] Processing {len(all_urls)} images through discovery filter funnel")

    try:
        biz_res = supabase.table("businesses") \
            .select("business_id, name, industry, business_type, currency") \
            .eq("business_id", business_id) \
            .single() \
            .execute()
        business_info = biz_res.data or {}
    except Exception:
        business_info = {}

    is_ecommerce        = business_info.get("business_type") == "ecommerce"
    existing_handles    = get_existing_handles(business_id)
    processed_hashes    = set()
    discovered_in_batch = []

    for index, media_url in enumerate(all_urls):
        print(f"  [ProductsHandler] [{index+1}/{len(all_urls)}] {media_url[:50]}...")

        # 1. Download image bytes
        image_bytes = await download_image(media_url)
        if not image_bytes:
            continue

        # Exact Duplicate Match (Group identical items instantly using binary hash)
        img_hash = hashlib.md5(image_bytes).hexdigest()
        if img_hash in processed_hashes:
            print(f"  [ProductsHandler] Exact duplicate binary found — skipping")
            continue
        processed_hashes.add(img_hash)

        # Upload image to storage bucket
        stored_url, file_path = upload_to_storage(business_id, run_id, image_bytes, index)
        if not stored_url:
            continue

        log_storage_usage(business_id, run_id, file_path, len(image_bytes))

        # 2. Thumbnail AI Filter (Run 1)
        is_prod_res = await filter_thumbnail_ai(stored_url, business_info, business_id, run_id)
        if not is_prod_res or not is_prod_res.get("is_product"):
            print(f"  [ProductsHandler] Not a valid product matching context — skipping")
            continue

        reason_desc = is_prod_res.get("reason", "")

        # 3. Semantic Deduplication / Alternate View Check (Run 2)
        if discovered_in_batch:
            is_dup_res = await check_semantic_duplicate(stored_url, reason_desc, discovered_in_batch, business_id, run_id)
            if is_dup_res and is_dup_res.get("is_duplicate"):
                print(f"  [ProductsHandler] Semantic match found (alternate image perspective) — skipping")
                continue

        # 4. High-Res Enrichment (Run 3 — Always Multiplied)
        ai_result = await analyse_image(stored_url, business_info, business_id, run_id)
        if not ai_result or ai_result.get("confidence", "low") == "low":
            continue

        # 5. Handle deduplication check against database records
        title  = ai_result.get("title") or "Unnamed Product"
        handle = build_handle(title)
        if handle in existing_handles:
            print(f"  [ProductsHandler] '{handle}' already exists in DB — skipping")
            continue

        # 6. Save product to database
        saved_handle = insert_product(business_id, ai_result, stored_url, is_ecommerce)
        if saved_handle:
            existing_handles.add(saved_handle)
            discovered_in_batch.append({
                "title":       title,
                "handle":      saved_handle,
                "description": ai_result.get("description_short") or ""
            })
            print(f"  [ProductsHandler] ✓ Discovered & Saved '{title}'")

        # Mark staging row as processed
        try:
            supabase.table("product_image_staging") \
                .update({"status": "processed", "processed_at": datetime.now(timezone.utc).isoformat()}) \
                .eq("business_id", business_id) \
                .eq("media_url", media_url) \
                .execute()
        except Exception as e:
            print(f"  [ProductsHandler] Staging status update failed: {e}")

    print(f"\n[ProductsHandler] ✓ Complete — {len(discovered_in_batch)} products saved as discovered")

# ============================================================
# SECTION 7 — FASTAPI APP
# ============================================================

app = FastAPI(title="ProductsHandler")


class ProcessImagesRequest(BaseModel):
    business_id:         str
    approved_media_urls: list[str] = []


@app.post("/process-images")
async def process_images(req: ProcessImagesRequest, background_tasks: BackgroundTasks):
    if not req.business_id.strip():
        raise HTTPException(status_code=400, detail="business_id is required")

    background_tasks.add_task(
        run_products_pipeline,
        req.business_id.strip(),
        req.approved_media_urls
    )
    return {
        "status":      "queued",
        "business_id": req.business_id,
        "image_count": len(req.approved_media_urls)
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "products-handler"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)
