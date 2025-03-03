from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from tqdm import tqdm

class judgement(BaseModel):
    Faithfulness: bool = Field(description="Faithfulness answer")
    Factuality: bool = Field(description="Factuality answer")
    # Rationale: str = Field(description="Rationale feedback")


def get_judgement(generated_text):
    generated_text = generated_text.replace("json", "").replace("`", "")
    try:
        parser = PydanticOutputParser(pydantic_object=judgement)

        formatted_output = parser.parse(generated_text)
        # print(formatted_output.Faithfulness, formatted_output.Factuality)
        return formatted_output.Faithfulness, formatted_output.Factuality
    except:
        test_list = generated_text.replace("True", "true").replace("False", "false").split()
        try:
            for i in range(len(test_list)):
                if test_list[i] in ["{", "{\"Faithfulness\":", "\"Faithfulness\":", "Faithfulness:", "\"Faithfulness\"", "Faithfulness"]:
                    generated_text = " ".join(test_list[i:])
                    break
            parser = PydanticOutputParser(pydantic_object=judgement)
            formatted_output = parser.parse(generated_text)
            return formatted_output.Faithfulness, formatted_output.Factuality
        except:
            try:
                for i in range(len(test_list)):
                    if test_list[i] in ["{", "\"Faithfulness\":", "Faithfulness:", "\"Faithfulness\"", "Faithfulness"]:
                        for j in range(i+1, i+4):
                            if test_list[j] in ["true,", "false,", "True,", "False,", "true", "false", "True", "False"]:
                                generated_text = "{" + " ".join(test_list[i:i+4]).strip(",") + "}"
                                break
                parser = PydanticOutputParser(pydantic_object=judgement)
                formatted_output = parser.parse(generated_text)
                return formatted_output.Faithfulness, formatted_output.Factuality
            except:
                # print("Error: ", generated_text)
                return -1, -1


def get_accuracy(data, CLabel_data, filter_fail):
    def transfor2label(sf, wf): 
        # if CLabel_data is not None:
            # bool, bool -> 0, 1, 2, -1 = fh, nfh, noh, error / unknown 
        if sf == False and wf == True:
            return 0
        elif sf == False and wf == False:
            return 1
        elif sf == True and wf == True:
            return 2
        else:
            return -1

    all_preds = []
    pred_fh, pred_nfh, pred_noh = [], [], []
    label_fh, label_nfh, label_noh = [], [], []
    
    for i in tqdm(range(len(data))):
        dict_result = data[i]


        # label = dict_result['label'] # factual_hall, non_factual_hallu, no_hallu, 
        if CLabel_data is not None:
            label = CLabel_data[i]['label']
             # generated_text = dict_result['wiki_evi_result']['generated_text']
            if isinstance(dict_result["llm_evi_result_zs"], dict):
                if "judge" in dict_result["llm_evi_result_zs"].keys(): # cot
                    generated_text = dict_result["llm_evi_result_zs"]['judge']
                else:# gpt-4o
                    generated_text = dict_result["llm_evi_result_zs"]['generated_text']


        sf, wf = get_judgement(generated_text)
        # print(sf, wf)
        all_preds.append([sf, wf])

        pred_label = transfor2label(sf, wf)
        # print(pred_label)
        if label == "factual_hallu":
            if not filter_fail or pred_label != -1:
                label_fh.append(0)
                pred_fh.append(pred_label)
        elif label == "non_factual_hallu":
            if not filter_fail or pred_label != -1:
                label_nfh.append(1)
                pred_nfh.append(pred_label)
        elif label == "no_hallu":
            if not filter_fail or pred_label != -1:
                label_noh.append(2)
                pred_noh.append(pred_label)


    avg_mode = "macro"
    acc_fh = accuracy_score(label_fh, pred_fh)
    acc_nfh = accuracy_score(label_nfh, pred_nfh)
    acc_noh = accuracy_score(label_noh, pred_noh)
    
    y_true = label_fh+label_nfh+label_noh
    y_pred = pred_fh+pred_nfh+pred_noh
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg_mode)  
    recall = recall_score(y_true, y_pred, average=avg_mode)
    f1 = 2 * precision * recall / (precision + recall)
    return acc_fh, acc_nfh, acc_noh, pred_fh, pred_nfh, pred_noh, label_fh, label_nfh, label_noh



def main_fhsumbench():
    CLabel_data_path = "origin_data_path"
    data_path = "generation_path"

    score_path = ""

    if_filter = False
    with open(CLabel_data_path, 'r') as f_in:
        CLabel_data = json.load(f_in)

    with open(data_path, 'r') as f_in:
        data = json.load(f_in)
        all_preds, acc_fh, acc_nfh, acc_noh, accuracy, precision, recall, f1, pred_fh, pred_nfh, pred_noh, label_fh, label_nfh, label_noh = get_accuracy(data, CLabel_data, if_filter)
        print(acc_fh, acc_nfh, acc_noh, accuracy, precision, recall, f1)

    scores = {"data_path": data_path, "acc_fh, acc_nfh, acc_noh": [acc_fh, acc_nfh, acc_noh], \
            "accuracy, precision, recall, f1": [accuracy, precision, recall, f1], \
            "pred_fh, label_fh": [pred_fh, label_fh], \
            "pred_nfh, label_nfh": [pred_nfh, label_nfh], \
            "pred_noh, label_noh": [pred_noh, label_noh],
            "all_preds": all_preds}
    with open(score_path, 'w') as f_out:
        json.dump(scores, f_out)




if __name__ == "__main__":
    main_fhsumbench()
