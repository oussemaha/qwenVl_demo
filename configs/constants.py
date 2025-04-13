MODEL_NAME="/home/ubuntu/Qwen2-VL-7B-Instruct"
SYSTEM_PROMPT_PATH="./configs/system_prompt.txt"
TEMP_DIR="./TEMP"
UNIVERSAL_KEYWORDS = [
    # English, French, Spanish, Italian, German
    "invoice", "bill", "total", "payment", "amount", "order",
    "facture", "devis", "paiement", "montant",  # French
    "fattura", "ricevuta", "pagamento", "importo","IVA"  # Italian
    "rechnung", "quittung", "zahlung", "betrag",  # German
    "factura", "recibo", "pago", "importe",  # Spanish
    # Arabic (فاتورة, إيصال, etc.)
    "فاتورة", "إيصال", "دفع", "المبلغ", 
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