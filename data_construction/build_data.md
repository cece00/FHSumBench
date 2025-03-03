'''
EntFA: 
1) remove non-factual hallucination entities in entfa -> the sentence will be either no-hallucinated or have factual hallucinations
2) NER and add entity description matching
    - ner.py and add_wiki.py
    - for entites with multiple descriptions, manually select the most relevant description.
    - remove mismatched descriptions for entities
3) add entity description to no-hallucinated samples (correct or randomly wrong)
    inject_hallu.py
    - inject factual and non-factual hallucination into the sentences


FactCollect:
1) NER
2) add entity description to no-hallucinated samples (correct or randomly wrong)
'''


