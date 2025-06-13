
import json
import spacy
from dateutil.parser import parse
from typing import Dict, Any



class JSONComparator:
    
    def __init__(self, 
                 llm_extractor,
                 nlp_model: str = "en_core_web_md",
                 text_similarity_threshold: float = 0.8):
        """
        Initialize with enhanced text comparison handling.
        """
        self.llm_extractor = llm_extractor
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Warning: {nlp_model} not found. Falling back to 'en_core_web_sm'")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.text_sim_threshold = text_similarity_threshold

    def _has_vectors(self, doc) -> bool:
        """Check if document has valid word vectors"""
        return any(token.has_vector for token in doc)

    def compare_text(self,text1: str, text2: str) -> bool:
    
        return self.llm_extractor.llm_compare_texts(text1, text2)

    def is_date(self, value: Any) -> bool:
        """Check if value is a parseable date string."""
        if isinstance(value, str):
            try:
                parse(value, fuzzy=False)
                return True
            except (ValueError, OverflowError):
                return False
        return False

    def compare_values(self, v1: Any, v2: Any) -> bool:
        """Compare values with type-specific rules"""
        # Handle None cases
        if v1 is None or v2 is None:
            return v1 == v2
            
        # Strict comparison for non-strings
        if not isinstance(v1, str) or not isinstance(v2, str):
            return v1 == v2
            
        # Date comparison
        if self.is_date(v1) and self.is_date(v2):
            return parse(v1).date() == parse(v2).date()
            
        # Text comparison
        return self.compare_text(v1, v2)

    def compare_jsons(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> Dict[str, bool]:
        """Compare two JSON objects field-by-field"""
        json2["REF_CONTRAT"]= json1["REF_CONTRAT"]
        comp= {key: self.compare_values(json1.get(key), json2.get(key)) 
                for key in json1}
        if "BUYER_ADDRESS" in comp and  not comp["BUYER_ADDRESS"]:
            comp["BUYER_ADDRESS"]=self.compare_text(json1["BUYER_ADDRESS"], json2["BUYER_COUNTRY"])
        if "SELLER_ADDRESS" in comp and not comp["SELLER_ADDRESS"]:
            comp["SELLER_ADDRESS"]=self.compare_text(json1["SELLER_ADDRESS"], json2["SELLER_COUNTRY"])
        return comp
        
# Example Usage
if __name__ == "__main__":
    # Initialize comparator with custom settings
    comparator = JSONComparator(
        text_similarity_threshold=0.75,  # More lenient text matching
        strict_numbers=False,            # Treat 5 and 5.0 as equal
        ignore_time_in_dates=True       # Compare only dates, not timestamps
    )

    json1 = {
        "name": "iPhone 15 Pro",
        "price": 999,
        "release_date": "2023-09-22",
        "description": "A premium smartphone."
    }

    json2 = {
        "name": "IPhone 15 pro",
        "price": 999.0,
        "release_date": "22nd September 2023",
        "description": "a good smartphone."
    }

    result = comparator.compare_jsons(json1, json2)
    print(json.dumps(result, indent=2))