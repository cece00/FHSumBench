import json



def remove():
    path = "path_to/EntFA/data/test.json"
    output_path = "path_to/EntFA/data/test_factual.json"

    with open(path, "r") as f, open(output_path, "w") as f_out:
        data = json.load(f) # list of dicts with lenth=240

        for i in range(len(data)):
            # print(data[i].keys()) # dict_keys(['source', 'reference', 'prediction', 'entities'])
            entities =  data[i]["entities"]
            rmv_index = []
            for j in range(len(entities)-1, -1, -1):  # Iterate from back to front
                if entities[j]["label"] == "Non-factual Hallucination":
                    # Get the start and end positions of the entity
                    start = entities[j]["start"]
                    end = entities[j]["end"]
                    text = data[i]["prediction"]
                    
                    # Calculate the total length to remove including trailing and leading spaces
                    left_spaces = len(text[:start]) - len(text[:start].rstrip())
                    right_spaces = len(text[end:]) - len(text[end:].lstrip())
                    actual_start = start - left_spaces
                    actual_end = end + right_spaces
                    removed_length = actual_end - actual_start -1

                    # Remove the entity and surrounding spaces from the text
                    text = text[:actual_start] +' ' + text[actual_end:]
                    data[i]["no_nonFact_prediction"] = text

                    # Update indices for all entities that come before the removed entity
                    for k in range(len(entities)-1, j, -1):
                        if entities[k]["start"] > actual_start:
                            entities[k]["start"] -= removed_length
                        if entities[k]["end"] > actual_start:
                            entities[k]["end"] -= removed_length
                    rmv_index.append(j)
            
            for r_j in rmv_index:
                entities.pop(r_j)
            data[i]["no_nonFact_entities"] = entities
        
        # write the updated data to the output file
        f_out.write(json.dumps(data,indent=4) + "\n")


def check_info():
    path = "../baselines_git/EntFA/data/test.json"

    with open(path, "r") as f:
        data = json.load(f) # list of dicts with lenth=240
        # print(data[0])
        ent_label_dic = {}
        for i in range(len(data)):
            ents = data[i]["entities"]
            for e in ents:
                if e["label"] not in ent_label_dic.keys():
                    ent_label_dic[e["label"]] = 1
                else:
                    ent_label_dic[e["label"]] += 1

check_info()