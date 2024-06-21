import openai
import numpy as np
import json

openai.api_key = "your key"
model = "text-embedding-3-small"


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def get_embedding(sentence, model="text-embedding-3-small"):
    """문장의 임베딩을 반환"""
    response = openai.Embedding.create(
        model=model,
        input=sentence
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

def add_similarity_scores_to_dataset(input_file, output_file):
    """데이터셋에 유사도 점수 추가"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 문장 쌍의 유사도 점수 계산
    for item in data:
        embedding1 = get_embedding(item['original'])
        embedding2 = get_embedding(item['paraphrase'])
        similarity_score = cosine_similarity(embedding1, embedding2)
        item['similarity_score'] = similarity_score
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 데이터셋 파일 경로 설정
input_file = '/data03/hyunaz/project/inner_ft/Finalparaphrases.json'
#input_file = '/data03/hyunaz/project/inner_ft/ff.json'
output_file = '/data03/hyunaz/project/inner_ft/FinalFinalparaphrases.json'

# 함수 실행하여 데이터셋에 유사도 점수 추가
add_similarity_scores_to_dataset(input_file, output_file)
