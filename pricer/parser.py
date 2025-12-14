from pricer.items import Item
import json
import re

# Settings – easy to change
MIN_CHARS = 600
MIN_PRICE = 0.50
MAX_PRICE = 999.99
MAX_TEXT_PER_PART = 3000
MAX_TOTAL_TEXT = 4000

# Things we never want in the description
JUNK_FIELDS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

# Clean up messy text (remove extra spaces, newlines, etc.)
def clean_text(text) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())[:MAX_TEXT_PER_PART]  # splits on whitespace → joins with single space

# Remove product codes like ABC12345, X1K2Z3K9P, etc.
def remove_product_codes(text: str) -> str:
    pattern = r"\b[A-Z0-9]{7,}\b"        # 7+ chars of letters/numbers
    pattern = r"\b(?=.*[A-Z])(?=.*\d)[A-Z0-9]{7,}\b"  # must have at least one letter + one number
    return re.sub(pattern, "", text)

# Build a nice clean description
def build_description(title, description, features, details) -> str:
    # Remove junk fields from details
    for junk in JUNK_FIELDS:
        details.pop(junk, None)

    parts = [title]

    if description:
        parts.append(clean_text(description))
    if features:
        parts.append(clean_text(features))
    if details:
        parts.append(json.dumps(details, ensure_ascii=False))

    full_text = "\n".join(parts)
    full_text = remove_product_codes(full_text)
    full_text = " ".join(full_text.split())          # final cleanup
    return full_text[:MAX_TOTAL_TEXT]

# Convert weight to pounds (returns 0 if missing or unknown)
def get_weight_in_pounds(details) -> float:
    weight_str = details.get("Item Weight", "")
    if not weight_str:
        return 0.0

    try:
        parts = weight_str.split()
        amount = float(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else ""

        if "pound" in unit:
            return amount
        if "ounce" in unit:
            return amount / 16
        if "gram" in unit:
            return amount / 453.592
        if "kilogram" in unit or "kg" in unit:
            return amount * 2.20462
        if "hundredth" in unit and len(parts) > 2 and "pound" in parts[2].lower():
            return amount / 100
    except:
        pass
    return 0.0

# Main function: turn raw Amazon data → clean Item (or None if invalid)
def parse(datapoint: dict, category: str):
    # Get and validate price
    try:
        price = float(datapoint.get("price", 0))
    except (ValueError, TypeError):
        return None

    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    # Load the JSON details safely
    try:
        details = json.loads(datapoint.get("details", "{}"))
    except:
        details = {}

    title = datapoint.get("title", "")
    description = datapoint.get("description")
    features = datapoint.get("features")

    # Build clean full description
    full_description = build_description(title, description, features, details)

    if len(full_description) < MIN_CHARS:
        return None

    weight = get_weight_in_pounds(details)

    # Return the clean Item
    return Item(
        title=title.strip(),
        category=category,
        price=price,
        description=full_description,
        weight=round(weight, 3),
    )