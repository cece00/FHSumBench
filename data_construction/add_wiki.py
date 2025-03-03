import json
import argparse
from tqdm import tqdm
import time


def get_alias_dict():
    alias_path = "path_to/wikidata5m_entity.txt"
    alias_dict = {}
    with open(alias_path, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            alias_dict[line[0]] = []
            for i in range(1, len(line)):
                alias_dict[line[0]].append(line[i].lower().strip())
    return alias_dict



def read_kg(kg_name):
    if kg_name == "wikidata":
        path = "path_to/kg/wikidata/entities.json"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # print(len(data)) # 77951
            # print(data["Q68128"])
            alias_dict = get_alias_dict()
            print("Load alias done.")
            
            ent_desc = {}
            for key, value in data.items():
                alias = []
                if key in alias_dict.keys():
                    alias = alias_dict[key]
                if value["label"].lower() not in ent_desc.keys():
                    ent_desc[value["label"].lower()] = [{"desc": value["description"], "id": key, "alias": alias}]
                else:
                    ent_desc[value["label"].lower()].append({"desc": value["description"], "id": key, "alias": alias})

        print("Load kg done.")
        return ent_desc



def add_kg_ent(ent_desc):
    path = "path_to/data_ner_merged.json"
    with open(path, 'r') as f:
        data = json.load(f)

        for i in tqdm(range(len(data))):
            item = data[i]
            for ent in item["no_nonFact_entities"]:
                ent_name = ent["ent"].lower()
                if ent_name in ent_desc.keys():
                    ent["desc"] = ""
                    if len(ent_desc[ent_name]) > 1:
                        # remove the empty description
                        for j in range(len(ent_desc[ent_name])-1, -1, -1):
                            if len(ent_desc[ent_name][j]["desc"]) == 0:
                                ent_desc[ent_name].pop(j)

                        # match the description
                        if len(ent_desc[ent_name]) == 1:
                            ent["desc"] = ent_desc[ent_name][0]["desc"]
                        # if the description is more than one
                        elif ent_name in ["india", "casablanca", "washington", "boston", "alexandria", "christian", "manhattan"]:
                            ent["desc"] = ent_desc[ent_name][0]["desc"]
                        elif ent_name in ["paris", "brazil", "europe"]:
                            ent["desc"] = ent_desc[ent_name][1]["desc"]
                        elif ent_name in ["chicago"]:
                            ent["desc"] = ent_desc[ent_name][2]["desc"]
                        elif "david" in ent_name or "glendale" in ent_name or "charlotte" in ent_name:
                            ent.pop("desc")
                            continue
                        elif len(ent_desc[ent_name]) > 1 and ent_name not in ["india", "casablanca", "washington", "boston", "paris", "brazil", "europe", "david", "glendale", "alexandria", "christian", "manhattan"]:
                            for j in range(len(ent_desc[ent_name])):
                                desc_item = ent_desc[ent_name][j]
                                if "United Kingdom" in desc_item["desc"] or "London" in desc_item["desc"] or "England" in desc_item["desc"] or "Europe" in desc_item["desc"]:
                                    ent["desc"] = desc_item["desc"]
                                    break

                        if len(ent["desc"]) == 0 and len(ent_desc[ent_name]) > 0:
                            ent["desc"] = ent_desc[ent_name]
                            ent["desc_tag"] = "multiple_desc" # for entities with multiple descriptions, manually select the most relevant description.
                    elif len(ent_desc[ent_name]) > 0:
                        ent["desc"] = ent_desc[ent_name][0]["desc"]
                
                else:
                    # find the entity description by alias
                    for k,v in ent_desc.items():
                        if_find = False
                        for v_i in v:
                            if ent_name in v_i["alias"]:
                                ent["desc"] = v_i["desc"]
                                if_find = True
                                break
                        if if_find:
                            break

        output_path = "path_to/tmp_data/data_ner_merged_kg.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def add_non_fact_desc(ent_desc):
    import random

    path = "path_to/tmp_data/data_ner_merged_kg.json"
    with open(path, 'r') as f:
        data = json.load(f)
        ent_num = 0
        for i in range(len(data)):
            entities = data[i]["no_nonFact_entities"]
            ent_num += len(entities)
        random.seed(100)
        all_rdm = random.sample(range(0, len(ent_desc)-1),ent_num)
        r_i = 0

        for i in tqdm(range(len(data))):
            entities = data[i]["no_nonFact_entities"]
            for ent in entities:
                if "desc" in ent.keys():
                    # find a non-fact description for the entity
                    rdm = all_rdm[r_i]
                    r_i += 1
                    cnt = 0 # the index of the entity description
                    for k,v in ent_desc.items():
                        if cnt == rdm and ent["ent"] != k and len(v[0]["desc"]) > 0:
                            ent["desc_nonFact"] = v[0]["desc"] 
                            break
                        elif cnt == rdm and (ent["ent"] == k or len(v[0]["desc"]) == 0):
                            if rdm == len(all_rdm)-1:
                                rdm = 0
                            else:
                                rdm += 1
                        cnt += 1
    output_path = "path_to/tmp_data/data_ner_merged_kg_nonFactdesc.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    rdm_path = "path_to/tmp_data/rdm_nonFactdesc.txt"
    with open(rdm_path, 'w') as f:
        json.dump(all_rdm, f, ensure_ascii=False, indent=4)

def stat():
    path = "path_to/tmp_data/data_ner_merged_kg_nonFactdesc.json"
    with open(path, 'r') as f:
        data = json.load(f)
        cnt_total = 0
        cnt_fact = 0
        cnt_nonFact = 0
        for i in range(len(data)):
            item = data[i]
            for ent in item["no_nonFact_entities"]:
                cnt_total += 1
                if "desc" in ent.keys():
                    cnt_fact += 1
                if "desc_nonFact" in ent.keys():
                    cnt_nonFact += 1
        # print("Total entities:", cnt_total)
        # print("Total entities with descriptions:", cnt_fact)
        # print("Total entities with non-fact descriptions:", cnt_nonFact)
        # print("Percentage of entities with descriptions:", cnt_fact/cnt_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, default="wikidata")
    args = parser.parse_args()

    kg_name = args.kg

    ent_desc = read_kg(kg_name) # with entities as keys
    # add_kg_ent(ent_desc)
    add_non_fact_desc(ent_desc)

    stat()
