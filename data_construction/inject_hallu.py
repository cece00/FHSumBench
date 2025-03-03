import json
from tqdm import tqdm
import copy


def insert_phrase_with_positions(original_text, entities, target_entity, phrase_to_insert):
    """
    Insert a phrase and update all entity positions
    
    Args:
        original_text (str): Original text
        entities (list): List of dictionaries containing entity information
        target_entity (dict): The entity after which to insert the phrase
        phrase_to_insert (str): The phrase to insert
    
    Returns:
        tuple: (Updated text, Updated entity list, Position information of inserted phrase)
    """
    entity_end = target_entity["end"]
    
    # 计算要插入的完整内容（包括逗号和空格）
    insert_content = f", {phrase_to_insert}, "
    
    # 插入短语
    before_space, after_space = 0, 0
    before_entity = original_text[:entity_end]
    if " " == before_entity[-1]:
        before_space = 1
    after_entity = original_text[entity_end:]
    if len(after_entity) > 0 and " " == after_entity[0]:
        after_space = 1
    new_text = before_entity.rstrip() + insert_content + after_entity.lstrip()
    
    # 计算插入内容的起始和结束位置
    phrase_start = entity_end + 2  # 加2是因为", "
    phrase_end = phrase_start + len(phrase_to_insert)

    # 更新所有实体的位置
    updated_entities = copy.deepcopy(entities)
    inserted_length = len(insert_content)
    updated_entities = update_entity_positions(updated_entities, entity_end, inserted_length-(before_space+after_space))
    
    # 创建插入短语的位置信息
    after_inserted_info = {
        "h_prediction": new_text,
        "h_start": phrase_start,
        "h_end": phrase_end,
        "h_phrase": phrase_to_insert,
        "updated_pos_entities": updated_entities
    }
        
    return after_inserted_info



def update_entity_positions(entities, inserted_start, inserted_length):
    """
    Update entity position information
    
    Args:
        entities (list): List of dictionaries containing entity information
        inserted_start (int): Insertion position
        inserted_length (int): Length of inserted content
    
    Returns:
        list: Updated entity list
    """
    for i in range(len(entities)):
        # 如果实体在插入点之后，更新其位置
        if entities[i]["start"] >= inserted_start:
            entities[i]["start"] += inserted_length
            entities[i]["end"] += inserted_length
    return entities


def inject_hallu():
    path = "path_to/tmp_data/data_ner_merged_kg_nonFactdesc.json"
    with open(path, 'r') as f:
        data = json.load(f)

    for i in tqdm(range(len(data))):
        data[i]["fh_prediction"] = []
        data[i]["nfh_prediction"] = []
        entities = data[i]["no_nonFact_entities"]
        if "no_nonFact_prediction" in data[i].keys(): # the correct summary is under "no_nonFact_prediction"
            original_text = data[i]["no_nonFact_prediction"]
        else:
            original_text = data[i]["prediction"]

        for ent in entities:
            if "desc" in ent.keys() and len(ent["desc"]) > 0 and len(ent["desc_nonFact"]) > 0:
                fh_dic = insert_phrase_with_positions(original_text, entities, ent, ent["desc"])
                data[i]["fh_prediction"].append(fh_dic)
            if "desc_nonFact" in ent.keys() and len(ent["desc"]) > 0 and len(ent["desc_nonFact"]) > 0:
                nfh_dic = insert_phrase_with_positions(original_text, entities, ent, ent["desc_nonFact"])
                data[i]["nfh_prediction"].append(nfh_dic)


    output_path = "path_to/tmp_data/data_fh_nfh_hallu.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    inject_hallu()






