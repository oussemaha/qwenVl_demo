MODEL_NAME="/home/ubuntu/Qwen2-VL-7B-Instruct"
SYSTEM_PROMPT_PATH="./configs/system_prompt_formatted.txt"
TEMP_DIR="./TEMP"
CLASSIFICATION_GPU=True
CLASSIFICATION_KEYWORD_THRESHOLD=3
CLASSIFICATION_MIN_TEXT_LENGTH=50
UNIVERSAL_KEYWORDS = [
    # English, French, Spanish, Italian, German
    "invoice", "bill", "total", "payment", "amount", "order","tax","shipping"
    "facture", "devis", "paiement", "montant", "TVA" # French
    "fattura", "ricevuta", "pagamento", "importo",  # Italian
    "rechnung", "quittung", "zahlung", "betrag",  # German
    "factura", "recibo", "pago", "importe",  # Spanish
    # Arabic (فاتورة, إيصال, etc.)
    "فاتورة", "إيصال", "دفع", "المبلغ", 
    "dollar","euro","usd",
]

# Universal regex patterns (amounts, dates, invoice numbers)
UNIVERSAL_PATTERNS = [
    r"\d+[\.,]\d{2}\s*(USD|EUR|€|¥|£|ريال|دينار)",  # Amounts
    r"\d{2}[/\-\.]\d{2}[/\-\.]\d{4}",  # Dates (DD/MM/YYYY)
    r"(invoice|facture|fattura|rechnung|فاتورة)\s*#?\s*[\d-]+",  # Invoice numbers
    r"(total|totale|gesamtbetrag|المبلغ)\s*[:=]?\s*\d+",  # Total fields
    r"order\s*(confirmation|no\.?)\s*[\d-]+",  # Order numbers
    r"customer\s*(no\.?|number)\s*[\d-]+",  # Customer numbers
]