import json
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm")

def for_entfa():
    path = "path_to/tmp_data/entfa_test_factual.json"
    with open(path, 'r') as f:
        data = json.load(f)

        for i in tqdm(range(len(data))):
            if "no_nonFact_prediction" in data[i].keys():
                text = data[i]["no_nonFact_prediction"]
                doc = nlp(text)
                for ent in doc.ents:
                    metaInfo = {
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "type": ent.label_,
                        "ent": ent.text
                    } 
                    data[i]["no_nonFact_entities"].append(metaInfo)

    output_path = "path_to/tmp_data/entfa_test_factual_ner.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def for_factcollect():
    path = "path_to/tmp_data/factcollect_test.json"
    with open(path, 'r') as f:
        data = []
        for lines in f.readlines():
            data.append(json.loads(lines))

        out_data = []
        for i in tqdm(range(len(data))):
            if data[i]["label"] == "CORRECT":
                text = data[i]["summary"]
                doc = nlp(text)
                for ent in doc.ents:
                    metaInfo = {
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "type": ent.label_,
                        "ent": ent.text
                    }
                    data[i]["no_nonFact_entities"].append(metaInfo)
                out_data.append(data[i])
    output_path = "path_to/tmp_data/factcollect_test_ner.json"
    with open(output_path, 'w') as f:
        json.dump(out_data, f, indent=4)




for_entfa()
for_factcollect()



