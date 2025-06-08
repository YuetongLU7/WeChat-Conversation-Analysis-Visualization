"""
Translation API wrapper for WeChat sentiment analysis using Youdao API
"""
import os
import time
import hashlib
import requests
from typing import Dict, List, Optional, Union

class TranslationResult:
    """Translation result class"""
    def __init__(self, translation: str, source_lang: str, target_lang: str):
        self.translation = translation
        self.source_lang = source_lang
        self.target_lang = target_lang

class BaseTranslator:
    """Base translator class"""
    def __init__(self):
        pass
    
    def translate(self, text: str, source: str = "zh-CN", target: str = "en-US") -> TranslationResult:
        """
        Translate text
        
        Args:
            text: Text to translate
            source: Source language code
            target: Target language code
            
        Returns:
            TranslationResult: Translation result object
        """
        raise NotImplementedError("Subclasses must implement translate method")
    
    def batch_translate(self, texts: List[str], source: str = "zh-CN", target: str = "en-US") -> List[TranslationResult]:
        """
        Batch translate texts
        
        Args:
            texts: List of texts to translate
            source: Source language code
            target: Target language code
            
        Returns:
            List[TranslationResult]: List of translation result objects
        """
        results = []
        for text in texts:
            results.append(self.translate(text, source, target))
        return results

class YoudaoTranslator(BaseTranslator):
    """Youdao translator class"""
    def __init__(self):
        super().__init__()
        self.app_key = "55c5227b502473c6"
        self.app_secret = "g3qJmSgqRuwVJr1oRmANuOoEU3URAiAv"
        self.url = "https://openapi.youdao.com/api"
        
        # Language code mapping
        self.lang_map = {
            "zh-CN": "zh-CHS",
            "en-US": "en",
            "fr": "fr"
        }
    
    def _encrypt(self, sign_str: str) -> str:
        """Generate sign for Youdao API"""
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(sign_str.encode('utf-8'))
        return hash_algorithm.hexdigest()
    
    def _truncate(self, q: str) -> str:
        """Truncate text for Youdao API"""
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size-10:size]
    
    def translate(self, text: str, source: str = "zh-CN", target: str = "en-US") -> TranslationResult:
        """
        Translate text using Youdao API
        
        Args:
            text: Text to translate
            source: Source language code
            target: Target language code
            
        Returns:
            TranslationResult: Translation result object
        """
        # If text is empty, return directly
        if not text.strip():
            return TranslationResult(
                translation=text,
                source_lang=source,
                target_lang=target
            )
        
        # Convert language codes
        from_lang = self.lang_map.get(source, source)
        to_lang = self.lang_map.get(target, target)
        
        # Prepare parameters
        salt = str(int(time.time() * 1000))
        curtime = str(int(time.time()))
        sign_str = self.app_key + self._truncate(text) + salt + curtime + self.app_secret
        sign = self._encrypt(sign_str)
        
        data = {
            'q': text,
            'from': from_lang,
            'to': to_lang,
            'appKey': self.app_key,
            'salt': salt,
            'sign': sign,
            'signType': 'v3',
            'curtime': curtime,
        }
        
        try:
            response = requests.post(self.url, data=data)
            result = response.json()
            
            # errorCode '0' means success
            if result.get('errorCode') == '0' and 'translation' in result:
                return TranslationResult(
                    translation=result['translation'][0],
                    source_lang=source,
                    target_lang=target
                )
            else:
                # Better error logging
                print(f"Youdao translation failed. API response: {result}")
                return TranslationResult(
                    translation=text,
                    source_lang=source,
                    target_lang=target
                )
        except Exception as e:
            print(f"Youdao API request failed with exception: {e}")
            return TranslationResult(
                translation=text,
                source_lang=source,
                target_lang=target
            )

# Global translator instance
_translator = None

def get_translator(use_google: bool = True) -> BaseTranslator:
    """
    Get translator instance
    
    Args:
        use_google: Whether to use Google translate (ignored, always uses Youdao)
        
    Returns:
        BaseTranslator: Translator instance
    """
    global _translator
    if _translator is None:
        _translator = YoudaoTranslator()
    return _translator 