from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex, Document #ServiceContext
from llama_index.core import Settings
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json
import logging
import requests

import openai
import os
import anthropic



import re
remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

@dataclass
class Evidence:
    text: str
    source: str  # 'document' or 'knowledge_base'
    support_score: int = -1
    relevance_score: int = -1
    node_id: str = None

@dataclass
class Claim:
    text: str
    evidences: List[Evidence] = None
    # doc_score: float = 0.0
    # knowledge_score: float = 0.0
    doc_score: int = -1
    knowledge_score: int = -1
    trajectory: Dict = None # record the node_id of evidence, how many evidence has been retrieved until this claim can be judged. e.g. {"doc":[], "kg":[]}




def use_vllm(prompt_sys, prompt_user, max_tokens, temperature, port):
    url = f"http://localhost:{port}/generate"
    conversation = [
        {"role": "system", "content": prompt_sys},
        {"role": "user", "content": prompt_user}
    ]
    input_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation]) + "\nAssistant: "

    payload = {
        "prompt": input_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, json=payload).json()['generated_text'].strip().lower()
    # print(response)
    return response




with open("../MyopenAIKey.txt", "r") as f:
    api_key = f.read().strip()
os.environ["OPENAI_API_KEY"] = api_key


def get_response_gpt(prompt_sys, prompt_user, max_tokens, temperature):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user}        
            ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    result = response.choices[0].message.content.strip()
    # print("result: ",result)
    return result




class ChainOfEvidence:
    def __init__(self, document: str, knowledge_base: List[str], model_name: str = "", port: int=8080):
        self.model_name = model_name
        self.port = port
        
        self.document = document
        self.knowledge_base = knowledge_base
        
        # Initialize service context
        # self.service_context = ServiceContext.from_defaults(llm=self.llm)
        # self.settings = Settings(llm=self.llm)
        # Settings.llm = self.llm
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002") # embed_batch_size=100
        
        # Create document and knowledge base indices
        self.doc_chunks = self.chunk_document()
        # doc_documents = [Document(text=chunk) for chunk in self.doc_chunks]
        doc_documents = [Document(
                text=chunk,
                metadata={'node_id': str(i)}  # 使用索引作为node_id
            ) 
            for i, chunk in enumerate(self.doc_chunks)
        ]
        self.doc_index = VectorStoreIndex.from_documents(doc_documents)
        
        # kb_documents = [Document(text=kb) for kb in knowledge_base]
        kb_documents = [Document(
                text=kb,
                metadata={'node_id': str(i)}  # 使用索引作为node_id
            ) 
            for i, kb in enumerate(self.knowledge_base)
        ]
        self.kb_index = VectorStoreIndex.from_documents(kb_documents)
        # print("kb_documents: ", len(kb_documents))
        # print("doc_documents: ", len(doc_documents))


    def llm_generate(self, prompt_sys: str, prompt_user: str, max_tokens: int, temperature: float) -> str:
        if "gpt" not in self.model_name and "claude" not in self.model_name:
            response = use_vllm(prompt_sys, prompt_user, max_tokens, temperature, self.port)
        elif "gpt" in self.model_name:
            response = get_response_gpt(prompt_sys, prompt_user, max_tokens, temperature)
        elif "claude" in self.model_name:
            response = get_response_claude(prompt_sys, prompt_user, max_tokens, temperature)
        
        return response



    def chunk_document(self, chunk_size: int = 200) -> List[str]:
        """Split document into manageable chunks."""
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
        return text_splitter.split_text(self.document)

    def generate_query(self, claim: Claim, missing_info: str = None) -> str:
        """Generate a search query based on claim and missing information."""
        if missing_info:
            prompt_sys = f"""Generate a specific search query to find evidence about the following claim. Start with the query, DO NOT reply any other information."""
            
            prompt_user = f"""
            focusing on the missing information: {missing_info}
            Claim: {claim.text}
            Query: """

        else:
            prompt_sys = f"""Generate a specific search query to find evidence about the following claim. Make the query focused and specific. Start with the query, DO NOT reply any other information."""
            
            prompt_user = f"""
            Claim: {claim.text}
            Query: """
        
        response = self.llm_generate(prompt_sys, prompt_user, max_tokens=30, temperature=0.7)
        # query = response.text.strip()
        query = response.strip()
        return query


    def match_evidence(self, claim: str, candidate: str) -> Tuple[float, float]:
        """Return support_score and relevance_score for a candidate evidence."""
        prompt_sys = "You are a helpful assistant."
        
        prompt_user = f"""Analyze how the evidence relates to the claim. DO NOT make any assumption, reasoning or use previous owned knowledge EXCEPT the evidence. Start with the answer, DO NOT reply any other information.
        Answer in the following format, split with "\n":
            Rationale: Give the rationale for the answer, within 30 tokens. Relevant means whether the evidence provides connection to the claim or contains the information of the claim. If Relevant is Yes, then make a judgement on relation.
            Relevant: Yes / No
            Relation: Support / Contradict

         Example1:
            Claim: New York is an eastern state in the United States of America.
            Evidence: New York is known for its iconic landmarks like the Statue of Liberty and bustling New York City.
            Answer:
                Rationale: Eventhough the evidence is about New York, it does not directly support the claim that "New York is an eastern state in the United States of America." Instead, it mentions iconic landmarks and features of New York, such as the Statue of Liberty and New York City, without addressing its geographical location or status as an eastern state. Therefore, the evidence is not relevant to the claim.
                Relevant: No
                Relation: None

        Example2:
            Claim: Regular exercise improves mental health.
            Evidence: Studies show that individuals who engage in physical activity report lower levels of stress and anxiety.
            Answer:
                Rationale: The evidence indicates a positive relationship between physical activity and reduced mental health issues, aligning with the claim.
                Relevant: Yes
                Relation: Support

        Example3:
            Claim: the pilot of the crashed jet killed himself.
            Evidence: Jaysh al-Islam, the larger of the two groups, posted footage online which it claimed showed the pilot being held after ejecting from the jet. The video, bearing Jaysh al-Islam's logo, showed an object engulfed in flames followed by an interview with the supposed pilot.
            Answer:
                Rationale: The evidence addresses the topic of the pilot living status. The evidence suggests the pilot survived and was captured, which is inconsistent with the claim of suicide.
                Relevant: Yes
                Relation: Contradict

        Now, please answer based on the claim and evidence:
            Claim: {claim}
            Evidence: {candidate}
            Answer: 
        """


        response = self.llm_generate(prompt_sys, prompt_user, max_tokens=80, temperature=0.7)
        # lines = response.text.strip().split('\n')
        lines = response.lower().strip().strip("answer:").strip("answer").strip().split('\n')
        # print("match_evidence response: ", response, "\n")
        logger.info(f"Evidence: {candidate}")
        logger.info(f"match_evidence response: {response}")
        
        relevance_text, support_text = "none", "none"
        try:
            # print(lines)
            for l_i in range(len(lines)):
                line_i = lines[l_i].strip()
                if line_i[:7] == "relevan":
                    if ":" in line_i:
                        relevance_text = line_i.split(':')[1].strip().strip(",").strip(".")
                    else:
                        relevance_text = line_i.split(' ')[-1].strip().strip(",").strip(".")
                if line_i[:8] == "relation":
                    if ":" in line_i:
                        support_text = line_i.split(':')[1].strip().strip(",").strip(".")
                    else:
                        support_text = line_i.split(' ')[-1].strip().strip(",").strip(".")
        except:
            relevance_text, support_text = "none", "none"


        # parse the support score and relevance score
        if support_text in ["support", "supported"]:
            support_score = 1
        elif support_text in ["contradict", "contradicted", "not supported", "not support"]:
            support_score = 0
        else:
            support_score = -1

        if relevance_text in ["yes", "relevant", "true"]:
            relevance_score = 1
        elif relevance_text in ["no", "irrelevant", "false", "not relevant"]:
            relevance_score = 0
        else:
            relevance_score = -1

        return support_score, relevance_score, support_text, relevance_text


    def get_retriever(self, index, processed_nodes: set):
    """
    Create a retriever that filters out already processed nodes
    
    Args:
        index: Original vector storage
        processed_nodes: Set of already processed node IDs
    """
        # 创建一个过滤函数
        def filter_func(doc: Document) -> bool:
            # 假设文档的metadata中包含node_id
            return doc.metadata.get('node_id') not in processed_nodes

        # 使用过滤函数创建retriever
        return index.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 1,
                "filter": filter_func
            }
        )


    def search_evidence(self, claim: str, query: str, source: str, processed_nodes: set) -> Tuple[List[Evidence], set]:
        """Search for evidence in either document chunks or knowledge base."""
        index = self.doc_index if source == 'document' else self.kb_index

        retriever = index.as_retriever(similarity_top_k=2)
         # 使用retriever搜索
        docs = retriever.retrieve(query)
        # print("docs: ", docs)
        # nodes = retriever.retrieve(query)
        
        evidences = []
        for doc in docs:
            # 更新已处理节点集合
            processed_nodes.add(doc.metadata['node_id'])
            support_score, relevance_score, support_text, relevance_text = self.match_evidence(claim, doc.text)
            if relevance_score == 1:
                evidences.append(Evidence(
                    text=doc.text,
                    source=source,
                    support_score=support_score,
                    relevance_score=relevance_score,
                    node_id=doc.metadata['node_id']
                ))
        logger.info(f"**** source: {source}. Evidence: {evidences}")

        return evidences, processed_nodes


            
    def process_summary(self, summary: str, claims: List[str]) -> List[Claim]:
        intermediate_gen = {}
        """Process entire summary and return evaluated claims."""        
        # print("claims:", claims)
        claim_objects = [Claim(text=claim) for claim in claims]
        logger.info(f"claims: {[claim.text for claim in claim_objects]}")
        
        # Generate all queries upfront
        queries = []
        for claim in claim_objects:
            query = self.generate_query(claim)
            queries.append(query)
        intermediate_gen["queries"] = queries
        all_miss = []

        # Process each claim with its corresponding query
        evaluated_claims = []
        for claim, query in zip(claim_objects, queries):
            processed_nodes_doc = set()
            processed_nodes_kb = set()
            doc_evidences = []
            kb_evidences = []

            # Determine scores
            current_doc_evi = -1
            current_kb_evi = -1
                        
            if claim.evidences is None:
                claim.evidences = []
            
            # Document evidence search
            logger.info(f"Search for document evidence for claim: {claim.text}")
            doc_evidences, processed_nodes_doc = self.search_evidence(claim.text, query, 'document', processed_nodes_doc)
            
            missing_info_prompt_sys = f"""What key information is missing from our current evidence to make a judgement about this claim? Please reply the missing information in the format of a sentence within 20 tokens. If there are multiple missing information, reply the most important one. DO NOT reply any other information."""
            missing_info_prompt_user = f"""
                Claim: {claim.text}
                Current Evidence: {[e.text for e in doc_evidences]}
                
                Missing Information:"""
            
            # Check document evidences
            for e in doc_evidences:
                logger.info(f"Document evidence: {e}")
                if e.support_score != -1:
                    current_doc_evi = e.support_score
                    break
                    
            if current_doc_evi in [0, -1]:
                # Knowledge base evidence search
                logger.info(f"Search for knowledge evidence for claim: {claim.text}")
                missing_info = self.llm_generate(missing_info_prompt_sys, missing_info_prompt_user, max_tokens=30, temperature=0.7).strip()
                # print("missing_info: ", missing_info)
                logger.info(f"missing_info: {missing_info}")
                missing_info1 = re.sub(remove_chars, "", missing_info)
                query = missing_info1
                kb_evidences, processed_nodes_kb = self.search_evidence(claim.text, query, 'knowledge_base', processed_nodes_kb)
                all_miss.append(query)
            
            else:
                # Knowledge base evidence search
                logger.info(f"Search for knowledge evidence for claim: {claim.text}")
                kb_evidences, processed_nodes_kb = self.search_evidence(claim.text, query, 'knowledge_base', processed_nodes_kb)
                all_miss.append("")
            
            # Check knowledge base evidences
            for e in kb_evidences:
                logger.info(f"Knowledge evidence: {e}")
                if e.support_score != -1:
                    current_kb_evi = e.support_score
                    break
            
            # Update claim with evidences and scores
            claim.evidences = doc_evidences + kb_evidences
            claim.doc_score = current_doc_evi
            claim.knowledge_score = current_kb_evi
            claim.trajectory = {
                "doc_chunk_id": list(processed_nodes_doc),
                "kb_chunk_id": list(processed_nodes_kb)
            }
            # print("claim.text ", claim.text)
            evaluated_claims.append(claim)
            logger.info(f"Claim evaluation complete - doc_score: {current_doc_evi}, kb_score: {current_kb_evi}")
        intermediate_gen["missing_info"] = all_miss
        
        return evaluated_claims, intermediate_gen
        


def read_data(file_path):
    # file_path = "../tmp_data/data_fh_nfh_hallu_shuffled.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data



if __name__ == "__main__":

    # document = "Your long document text here..."
    # knowledge_base = ["Knowledge fact 1", "Knowledge fact 2", ...]
    # summary = "Summary to be evaluated..."

    import argparse
    from datetime import datetime
    import time
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y%m%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    suf = args.suffix
    start = args.start

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f'../logs/chain_cove_{suf}_{currentTime}.log')
    logger = logging.getLogger(__name__)

    model_name = args.model
    model_name_str = model_name.split("/")[-1]


    data_path = ""
    result_path = ""
    
    # gpt-4o generated claims
    claim_path = ""
    all_claims = read_data(claim_path)

    for i in tqdm(range(len(data))):

        claims = all_claims[i]["claims"]
        document = data[i]["document"]
        knowledge_base = []

        
        # wiki
        if "wiki" in suf:
            for k,v in data[1]["wiki_evi"].items():
                if len(v)>0 and len(v[0])>0:
                    knowledge_base.append(k + ": " + v[0])
        else:
            # llm
            for k, v in data[i]["llm_evi"].items():
                if len(v)>0:
                    knowledge_base.append(k + ": " + v)

        summary = data[i]["prediction"]
        logger.info(f"summary: {summary}")

        chain = ChainOfEvidence(document, knowledge_base, model_name=model_name, port=args.port)
        evaluated_claims, intermediate_gen = chain.process_summary(summary, claims)

        result = [{"doc_kb_scores": [c.doc_score, c.knowledge_score], "claim": c.text, "trajectory": c.trajectory} for c in evaluated_claims]


        data[i]["rag_llm_evi_result"] = result
        data[i]["intermediate_gen"] = intermediate_gen

        if "gpt" in model_name or "claude" in model_name:
            time.sleep(3)

        if i%5 == 0:
            with open(result_path, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                logger.info(f"save_file_to_i: {i}")

    with open(result_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)