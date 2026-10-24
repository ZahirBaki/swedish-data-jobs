"""Versioned Step 3E prompt on the unchanged budgeted Step 3D transport."""
import json
import re

from src.llm_classifier import LLMClassifier, VALIDATION_VERSION, stable_hash
from src.classify_jobs_v2 import CATEGORIES
from src.validate_classification import ROOT

PROMPT_VERSION = 'step3e-1.0'
POLICY_PATH = ROOT/'docs'/'classification_policy_step3e_v1.md'
PROMPT_PATH = ROOT/'prompts'/'step3e-v1.txt'


def multi_role_candidate(title, description):
    """Conservative review cue, not a verified classification/vacancy count."""
    title = title.casefold()
    return bool(re.search(r'data\s+(?:architects?|engineers?|scientists?).*(?:/|\band\b|\boch\b).*'
                          r'(?:architects?|engineers?|scientists?)',title)
                or re.search(r'(?:following positions|följande tjänster)\s*:',description.casefold()))


class Step3EClassifier(LLMClassifier):
    def __init__(self,*args,prompt_text=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.prompt_text=prompt_text if prompt_text is not None else PROMPT_PATH.read_text(encoding='utf-8')
        self.system_prompt=(self.prompt_text+'\nCANONICAL CATEGORIES:\n'+json.dumps(CATEGORIES,ensure_ascii=False)
                            +'\nPOLICY:\n'+self.policy_text)

    def cache_key(self,title,description):
        return stable_hash(dict(stage='step3e',prompt_version=PROMPT_VERSION,
            validation_version=VALIDATION_VERSION,policy_hash=self.policy_hash,
            provider=self.config.provider,endpoint=self.config.base_url.rstrip('/'),
            payload=self.payload(title,description)))

    def metadata(self,*args,**kwargs):
        result=super().metadata(*args,**kwargs)
        result.update(ai_prompt_version=PROMPT_VERSION,ai_policy_version='step3e-v1')
        return result

    def classify(self,identifier,title,description):
        result=super().classify(identifier,title,description)
        result.update(ai_prompt_version=PROMPT_VERSION,ai_policy_version='step3e-v1',
                      ai_multi_role_review_candidate=multi_role_candidate(title,description))
        if result['ai_multi_role_review_candidate']:
            result['ai_review_required']=True
        if self.execute and str(identifier) in self.state['results']:
            self.state['results'][str(identifier)]=result
            self.save()
        return result
