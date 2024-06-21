# app.py (Flask 백엔드)
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import time
import re
from vllm import LLM, SamplingParams
import pickle
import numpy as np
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from konlpy.tag import Okt, Komoran
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


app = Flask(__name__)

selected_model = '/data03/hyunaz/project/inner_ft/EEVE-Korean-Instruct-10.8B-v1.0' #Your model path
#selected_model = "/data03/hyunaz/project/inner_ft/eeve-nuclear-trans-v2-export"
#embedding_model = SentenceTransformer("/data03/hyunaz/project/inner_ft/output/kor_sts_-data03-hyunaz-project-inner_ft-ko-sroberta-multitask-2024-05-05_18-43-49")
embedding_model = SentenceTransformer("/data03/hyunaz/project/inner_ft/ko-sroberta-multitask") #Your embedding model path
global df

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

###load the preprocessed files 
with open('/data03/hyunaz/project/inner_ft/embeddingrr.pkl', 'rb') as f:
    df_500 = pickle.load(f)
with open('/data03/hyunaz/project/inner_ft/processed_file/bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)
# 데이터프레임의 열 이름 확인

#vllm apply
modelg = LLM(
    model=selected_model,
    #tokenizer=selected_model,
    tensor_parallel_size= 1,
    dtype='bfloat16',
    gpu_memory_utilization = 0.40,
    )
    
tokenizer = AutoTokenizer.from_pretrained(selected_model)  # 토크나이저 로드
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
global sampling_params
sampling_params = SamplingParams(
    temperature=0.5, 
    top_p=0.95,
    top_k=50,
    max_tokens= 3000,
    min_p=0.5,
    )

@app.route('/')
def index():
    return render_template('trans.html')

@app.route('/law')
def indexlaw():
    return render_template('law.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text_to_translate = data['text'] 
    text = data['text']
    model_prompt = f'''
    <Instruction> You are a competent translator. If the sentence in English is the input value, translate it into Korean and vice versa.
        '''  # 모델에 전달할 프롬프트 설정
    prompt_text = model_prompt + text


    translated_text = generate_response_with_prompt(text_to_translate, model_prompt)

    return jsonify({
        'translated_text': translated_text
    })

@app.route('/law_process', methods=['POST'])
def law_process():
    data = request.get_json()
    query_text = data['text']
    # Tokenize and encode the query for embedding generation
    query_embedding = embedding_model.encode(query_text)
    result = search_embeddings1(df_500,query_embedding,query_text)
    #print(result)
    text_to_process = result.iloc[0]['text'] if not result.empty else "No data available"
    #print(text_to_process)
    model_prompt = f"""
        <Instructions>
        Your task is to peruse and comprehend regulatory documents from the Korea Atomic Energy Research Institute (KAERI) and answer related questions.
        The question is listed in <Question>.
        The regulatory documents are listed in <Documents>.

        Follow the requirements unconditionally.
        They are listed in <Requirements>.
        Answers should follow the <styles>.
        <Requirements>
        you should find the answer to the question in the documents provided.
        Return a accurate and succinct answer based on the documents.
        </Requirements>
        
        <Question>
        """ + str(query_text) + """
        </Question>
        
        <Documents>
        (Note: The word in square brackets is the title of each document.)
                        
        1.""" + str(result.iloc[0]['text']) + """
        2.""" + str(result.iloc[1]['text']) + """
        </Documents>
        
        <styles>
        Do not generate creative answer. Ask for clarification if a user request is ambiguous.
        Use at most 700 characters or less.
        You are required to respond in Korean.
        Use a polite and formal tone in your response.
        </styles>
        """

    #law_QA = generate_response_with_prompt(model_prompt,result)
    law_QA = generate_response_with_prompt(text_to_process,model_prompt)
    return jsonify({
        'law_QA': law_QA
    })

def search_embeddings1(df_500, query_embedding, query_text, n=2):
    komoran = Komoran()
    
    def cosine_similarity_parallel():
        query_embedding_tensor = torch.tensor(query_embedding).unsqueeze(0)
        embeddings_tensor = torch.tensor(df_500.iloc[:, 3:771].values).float()
        cos_similarities = util.pytorch_cos_sim(query_embedding_tensor, embeddings_tensor)[0]
        df_500['similarity'] = cos_similarities.numpy()
        return df_500.sort_values('similarity', ascending=False).head(n)

    def bm25_parallel():
        def tokenizer_k(text):
            return komoran.morphs(text)
        
        tokenized_query = tokenizer_k(query_text)
        #print("Tokenized Query:", tokenized_query)
        scores = bm25.get_scores(tokenized_query)
        #print("BM25 Scores:", scores)

        if len(scores) == 0:
            return pd.DataFrame()  # Return an empty DataFrame if no scores
        
        top_n = np.argsort(scores)[::-1][:n]
        top_n = [i for i in top_n if i < len(df_500)]  # Ensure indices are within bounds
        #print("Top N Indices:", top_n)

        if len(top_n) == 0:
            return pd.DataFrame()  # Return an empty DataFrame if top_n is empty
        
        return df_500.iloc[top_n]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cosine = executor.submit(cosine_similarity_parallel)
        future_bm25 = executor.submit(bm25_parallel)
        results_co = future_cosine.result()
        results_bm = future_bm25.result()

    if results_co.empty and results_bm.empty:
        return pd.DataFrame()  # Return an empty DataFrame if both results are empty

    df_all = pd.concat([results_co, results_bm])
    df_all.drop_duplicates(subset=['text'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all.head(n)


def clean_spaces(token):
# 공백 정제 로직, 예를 들어 연속된 공백을 하나로 줄임
    return re.sub(r'\s+', ' ', token)


def generate_response_with_prompt(text, prompt):
    if not isinstance(prompt, str) or not isinstance(text, str):
        raise ValueError("Both 'prompt' and 'text' must be strings.")
    
    # Combine prompt and text, then tokenize
    prompt_text = prompt + text
    encoding = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=3000)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Generate response using the LLM model
    output = modelg.generate(prompts=prompt_text, sampling_params=sampling_params, prompt_token_ids=input_ids.tolist())
    output = output[0].outputs[0].token_ids  # Extract token ids from the generated output

    # Decode the generated tokens to text, clean it, and return
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    generated_text_cleaned = clean_spaces(generated_text)
    return generated_text_cleaned

if __name__ == '__main__':
    app.run(debug=False, port=7777)