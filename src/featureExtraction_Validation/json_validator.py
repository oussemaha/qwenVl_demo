
from dateutil.parser import parse
from typing import Dict, Any



class JSONComparator:
    
    def __init__(self, 
                 llm_extractor,
                 text_similarity_threshold: float = 0.8):
        """
        Initialize with enhanced text comparison handling.
        """
        self.llm_extractor = llm_extractor

        
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
    
    
    def comp_code_del_reg(self,v_user: Any, v_llm: Any) -> bool:
        list_code_delai_reg_similarity=[[13,33],[66,15],[12,22],[11,98],[11,18],[11,97],[11,98]]
        if v_user==v_llm:
            return True
        elif any(v_user in sublist and v_llm in sublist for sublist in list_code_delai_reg_similarity):
            return True
        elif (v_llm in [14,17] and v_user == 66) or (v_llm == 16 and v_user == 22):
            return True
        else:
            return False
        

    def compare_values(self, v_user: Any, v_llm: Any, key: str) -> bool:
        """Compare values with type-specific rules"""
        if key=="CODE_DELAI_REGLEMENT":
            return self.comp_code_del_reg(v_user,v_llm)

        # Handle None cases
        if v_user is None or v_llm is None:
            return v_user == v_llm
        
        # Date comparison
        if self.is_date(v_user) and self.is_date(v_llm):
            return parse(v_user).date() == parse(v_llm).date()
            
        # Strict comparison for non-strings
        if not isinstance(v_user, str) or not isinstance(v_llm, str):
            return v_user == v_llm
            
        
            
        # Text comparison
        return self.compare_text(v_user, v_llm)

    def compare_jsons(self, json_user: Dict[str, Any], json_llm: Dict[str, Any]) -> Dict[str, bool]:
        """Compare two JSON objects field-by-field"""
        json_llm["REF_CONTRAT"]= json_user["REF_CONTRAT"]
        comp= {key: self.compare_values(json_user.get(key), json_llm.get(key),key) 
                for key in json_user}
        try:
            if "BUYER_ADDRESS" in comp and  not comp["BUYER_ADDRESS"]:
                comp["BUYER_ADDRESS"]=self.compare_text(json_user["BUYER_ADDRESS"], json_llm["BUYER_COUNTRY"])
            if "SELLER_ADDRESS" in comp and not comp["SELLER_ADDRESS"]:
                comp["SELLER_ADDRESS"]=self.compare_text(json_user["SELLER_ADDRESS"], json_llm["SELLER_COUNTRY"])
        except KeyError:
            print("KeyError: One of the keys is missing in the JSON objects.")
        return comp
        