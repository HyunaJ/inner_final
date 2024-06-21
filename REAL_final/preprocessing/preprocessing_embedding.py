import pdfplumber
import pandas as pd
import os, re, tenacity, pickle, unicodedata
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor # 유사도 계산 병렬 처리를 위한 라이브러리.
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity # 텍스트 임베딩 API, 문장 간 유사성 계산
import json
from datetime import datetime
openai.api_key = "your key"
okt = Okt() # Okt 클래스의 객체를 생성하여 한글 형태소 분석을 수행할 수 있도록 함

# 원숫자 ① -> 1., o->ㅇ, △-> ㅁ
def circled_to_normal(char):
    circled_numbers = u"\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469\u25CB\u25B3"
    normal_numbers = "1234567890ㅇㅁ"
    
    if char in circled_numbers[:-2]:
        return normal_numbers[circled_numbers.index(char)] + "."
    if char in circled_numbers[-2:]:
        return normal_numbers[circled_numbers.index(char)]
    return char

class Chatbot():
    
    # PDF 텍스트 분석 및 처리
    def parse_paper(self, pdf):
        print("Parsing paper")
        number_of_pages = len(pdf.pages)
        print(f"Total number of pages: {number_of_pages}")
        paper_text = []
        blob_text = ''
        
        # 페이지 텍스트 추출
        for i in range(number_of_pages):
            page_lines = ''
            page = pdf.pages[i]

            def visitor_body(text, cm, tm, fontDict, fontSize):
                global page_lines
                # lines = text.split('\n')
            page_lines = page.extract_text(visitor_text=visitor_body)
            blob_text += page_lines
            
        blob_text = re.sub(r'\n', '', blob_text)
        blob_text = "".join([circled_to_normal(c) for c in blob_text]) # 원문자, ○△ 변경
        blob_text = blob_text.replace('\u0027', '') # '
        blob_text = blob_text.replace('\u201C', '') # “ 
        blob_text = blob_text.replace('\u201D', '') # ”
        blob_text = blob_text.replace('\uFF05', '%') # ％ -> %
        blob_text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9!@#$%￦^&*(),.-~:{}<>/\\]', ' ', blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        
        # 목차 부분 삭제 -----
        blob_text = re.sub(r"재1조", r'제1조', blob_text) # 대학생실험실습인턴십지침 오타!!!
        blob_text = re.sub(r"재1장", r'제1장', blob_text)

        blob_text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", blob_text)
        blob_text = re.sub(r"제\s*(\d+)\s*장", r"제\1장", blob_text)
        
        pattern1 = re.compile(r'목차[\s\S]*제1장') # '목차'로 시작하고 '제1장'이 나오는 패턴
        blob_text = pattern1.sub('제1장', blob_text)

        start = blob_text.find('목차') # '목차'로 시작하고, '제1조'이 나오는 패턴
        # match = re.search(r'(제1조\([^)]+\)).*?(제1조\([^)]+\))', blob_text)
        match = re.search(r'(제1조).*?(제1조)', blob_text)
        
        if start != -1 and match:
            end = match.start(2)
            blob_text = blob_text[:start] + blob_text[end:]
        else:
            pass

        blob_text = re.sub(r'\([^)]*\)', ' ', blob_text) # (괄호 내용) 삭제
        blob_text = re.sub(r'\<[^)]*\>', ' ', blob_text) # <내용> 삭제
        
        # blob_text = blob_text.replace(' ','') # 띄어쓰기 없애기
        
        # # 형태소 분석 결과를 이용해 띄어쓰기를 적용한 문자열 생성
        # spacing = Spacing(rules=[title])
        # blob_text = spacing(blob_text)

        
        # 패턴 적용해서 결과 출력
        if blob_text[18:24] == '원규관리규정':
            blob_text = re.sub(r'부칙부 칙.*', '', blob_text)
            blob_text = re.sub(r"부 칙", r"부칙", blob_text)
        else:
            blob_text = re.sub(r"부 칙", r"부칙 ", blob_text)
            blob_text = re.sub(r'부칙 .*', '', blob_text) # 부칙 ~ 별칙 삭제 -> 원규관리규정 문서에서 적용xx.....
        # blob_text = re.sub(r"부 칙", r"부칙", blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\. ?', r'\1. ', blob_text) # 마침표 붙이기
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) +\, ?', r'\1, ', blob_text) # , 붙이기 
        # 문서 번호 및 날짜 
        pattern3 = r"\d{4}/\d{2}/\d{2}"
        blob_text = re.sub(f".*{pattern3}", "", blob_text)
        blob_text = re.sub(r"목차", r"", blob_text)
        blob_text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣]) (\의 |\로써 |\한다 |\은 |\을 |\를 |\이란 |\란 |\에 |\는 |\가 |\이라 |\라 |\이고 |\로 |\에서 |\에도 |\에게 |\야 |\과 )', r'\1\2', blob_text)
        blob_text = re.sub(r"결 산", r'결산', blob_text)
        blob_text = re.sub(r"기록 부", r'기록부', blob_text)
        blob_text = re.sub(r"재물조사 서", r'재물조사서', blob_text)
        blob_text = re.sub(r"절 차", r'절차', blob_text)
        blob_text = re.sub(r"소 정", r'소정', blob_text)
        blob_text = re.sub(r"연 구원", r'연구원', blob_text)
        blob_text = re.sub(r"석 사", r'석사', blob_text)
        blob_text = re.sub(r"박 사", r'박사', blob_text)
        blob_text = re.sub(r"정 함", r'정함', blob_text)
        blob_text = re.sub(r"체 계적인", r'체계적인', blob_text)
        blob_text = re.sub(r"계 획", r'계획', blob_text)
        blob_text = re.sub(r"보 증", r'보증', blob_text)
        blob_text = re.sub(r" 조 직 ", r' 조직 ', blob_text)
        blob_text = re.sub(r"여 부", r'여부', blob_text)
        blob_text = re.sub(r"채 용", r'채용', blob_text)
        blob_text = re.sub(r"승 진", r'승진', blob_text)
        blob_text = re.sub(r"급 여", r'급여', blob_text)
        blob_text = re.sub(r"겸 직", r'겸직', blob_text)
        blob_text = re.sub(r"복 무", r'복무', blob_text)
        blob_text = re.sub(r"삭 제", r'삭제', blob_text)
        blob_text = re.sub(r"보 칙", r'보칙', blob_text)
        blob_text = re.sub(r"교 육", r'교육', blob_text)
        blob_text = re.sub(r"승 격", r'승격', blob_text)
        blob_text = re.sub(r"구 성 원", r'구성원', blob_text)
        blob_text = re.sub(r"요 구", r'요구', blob_text)
        blob_text = re.sub(r"포 함", r'포함', blob_text)
        blob_text = re.sub(r"삭제", r'', blob_text)
        pattern4 = r'\.(\d+)\.' # .1. -> . 1. 띄우기
        blob_text = re.sub(pattern4, r'. \1.', blob_text)
        pattern5 = r'\.([ㄱ-ㅎㅏ-ㅣ가-힣])' # .+한글 띄기
        blob_text = re.sub(pattern5, r'. \1', blob_text)
        pattern6 = r'([ㄱ-ㅎㅏ-ㅣ가-힣])(\d+)\.' # 한글+숫자. 띄기
        blob_text = re.sub(pattern6, r'\1 \2.', blob_text)
        pattern7 = r'\.(제)' # 다.제1조 -> 다. 제1조
        blob_text = re.sub(pattern7, r'. \1', blob_text)
        pattern8 = r'([ㄱ-ㅎㅏ-ㅣ가-힣])(제\d+)' # 경우제1 -> 경우 제1
        blob_text = re.sub(pattern8, r'\1 \2', blob_text)
        blob_text = re.sub('-', ' - ', blob_text)
        blob_text = re.sub(r'\)', '', blob_text)
        blob_text = re.sub(r'\(', '', blob_text)
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        pattern9 = r'(\d+)\. (\d+)' # 숫자+.+숫자 붙이기
        blob_text = re.sub(pattern9, r'\1.\2', blob_text)
        pattern10 = r"(제\d+조)(\d+)" # 제(숫자)조1 -> 제(숫자)조 1
        blob_text = re.sub(pattern10, r"\1 \2 ", blob_text)
        pattern11 = r"(제\d+호의)(\d+)" # 제(숫자)호의1 -> 제(숫자)호의 1
        blob_text = re.sub(pattern11, r"\1 \2 ", blob_text)
        blob_text = re.sub(r'\.{2,}', '. ', blob_text) ## .. -> .
        blob_text = re.sub(r'\. +\.', '. ', blob_text) ## . . -> .
        blob_text = re.sub(r'\s{2,}', ' ', blob_text) # 공백 두 칸 이상 -> 한 칸으로
        
        if blob_text[-1] == ' ': 
            blob_text = blob_text[:-1]
        if blob_text[-2:] == '부칙': 
            blob_text = blob_text[:-2]

        paper_text = blob_text
        print("Done parsing paper")
        # print(paper_text)
        return paper_text


    def paper_df(self, pdf):  # pdf를 분석하여 df로 변환
        print('Creating dataframe')

        # pdf를 제목(제x조, 제x장)으로 분리하여 리스트에 저장
        sentences = re.split(r'(?<=\s)(?=제\d+조\s|제\d+장\s)', pdf)
        # 제목이 아닌 문장만 추출하여 리스트에 저장
        sentences = [s.strip() for s in sentences if not re.match(
            r'^제\d+조\s*$', s) and not re.match(r'^제\d+장\s*$', s.strip())]

        ss = ''  # 250자 이하의 문장들을 모아둘 변수
        ss2 = []  # 250자 이상의 문장들을 저장할 리스트

        for s in sentences:
            s = s.strip()
            if len(s) >= 400:  # s가 400글자 이상일 때
                if len(ss) > 250:  # 이전까지 모아진 ss가 있다면 ss2에 추가
                    ss2.append(ss.strip())
                    ss = ''
                    ss2.append(s)  # s도 추가
                else:  # ss가 250이하라면
                    if ss2 != []:
                        if len(ss2[-1]) < len(s):  # 작은 곳에 ss추가
                            ss2[-1] += ' ' + ss.strip()
                            ss = ''
                            ss2.append(s)
                    else:
                        ss2.append(ss.strip() + ' ' + s)  # s를 ss2에 추가
                        ss = ''
            else:  # s가 400글자 미만일 때
                if len(ss) > 250:
                    ss2.append(ss.strip())
                    ss = s
                else:
                    ss += (' ' + s)

        if len(ss) > 0:
            if ss2 != []:
                if len(ss) < 250:
                    ss2[-1] += ' ' + ss.strip()
                else:
                    ss2.append(ss.strip())
            else:
                ss2.append(ss.strip())

        # ss2를 데이터프레임으로 변환하여 반환
        df = pd.DataFrame(ss2)

        return df

#######################################################################  

def split_text(text, num_parts):
    sentences = re.split(r'(?<=다\.)\s', text) # '다. '로 쪼개기
    # sentences = re.split(r'(?<=\d\.)\s', text) # 번호로 쪼개기.

    part_length = len(text) // num_parts
    parts = []
    part = ''

    for sentence in sentences:
        part += sentence + ' '
        if len(part) >= part_length and len(parts) < num_parts - 1:
            parts.append(part.strip())
            part = ''

    parts.append(part.strip())
    return parts


def split_df(df):
    long_text_indices = df[df['text'].str.len() >= 600].index

    new_df = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        if index in long_text_indices:
            text_length = len(row['text'])
            
            if text_length >= 1200:
                num_parts = 6 #5
            elif text_length >= 1000:
                num_parts = 5 #4
            elif text_length >= 800:
                num_parts = 4 #4
            else:
                num_parts = 2
            
            parts = split_text(row['text'], num_parts)

            new_parts = []
            current_part = parts.pop(0) # 현재 처리 중인 청크
            while parts:
                next_part = parts.pop(0) #다음에 처리할 청크
                if len(current_part) + len(next_part) <= 300: #current_part와 next_part의 합이 300자를 초과하지 않는다면, 두 청크는 합쳐져 하나의 청크로 처리
                    current_part += ' ' + next_part
                else:
                    new_parts.append(current_part)
                    current_part = next_part

            new_parts.append(current_part)

            for part in new_parts:
                new_row = row.copy()
                new_row['text'] = part
                #new_df = new_df.append(new_row, ignore_index=True)
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            #new_df = new_df.append(row, ignore_index=True)
            new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
            
    new_df.reset_index(drop=True, inplace=True)
    new_df2 = new_df.copy()
    
    # 250자 이하 텍스트를 합치는 코드
    short_text_indices = new_df2[new_df2['text'].str.len() <=100].index #200

    final_df = pd.DataFrame(columns=df.columns)

    for index, row in new_df2.iterrows():
        if index in short_text_indices:
            prev_row = None
            next_row = None
            
            if index != 0: # 첫 번째 행이 아니면,
                prev_row = new_df2.iloc[index - 1]
                if index + 1 < len(new_df2): # 마지막 행도 아니면
                    next_row = new_df2.iloc[index + 1]
                    
                    if len(prev_row['text']) <= len(next_row['text']): # 이전 행과 이후 행 길이 비교
                        prev_row['text'] += ' ' + row['text']
                        row['text'] = ''
                        final_df = final_df.iloc[:-1:]
                        #final_df = final_df.append(prev_row, ignore_index=True)
                        final_df = pd.concat([final_df, pd.DataFrame([prev_row])], ignore_index=True)

                    else:
                        row['text'] = row['text'] + ' ' + next_row['text']
                        new_df2.iloc[index+1,0] = ''
                        #final_df = final_df.append(row, ignore_index=True)
                        final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
                else: # 마지막 행이면
                    prev_row['text'] += ' ' + row['text']
                    row['text'] = ''
                    final_df = final_df.iloc[:-1:]
                    #final_df = final_df.append(prev_row, ignore_index=True)
                    final_df = pd.concat([final_df, pd.DataFrame([prev_row])], ignore_index=True)
            else: # 첫 번째 행이면
                if index + 1 < len(new_df2): # 마지막 행이 아니라면
                    next_row = new_df2.iloc[index + 1]
                    row['text'] = row['text'] + ' ' + next_row['text']
                    new_df2.iloc[index+1,0] = ''
                    #final_df = final_df.append(row, ignore_index=True)
                    final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
                else: # 행 자체가 하나라면
                    pass
  
        else:
            #final_df = final_df.append(row, ignore_index=True)
            final_df = pd.concat([final_df, pd.DataFrame([row])], ignore_index=True)
            
    final_df = final_df[final_df['text'].str.strip() != '']
    final_df.reset_index(drop=True, inplace=True)

    # 700글자 이상인 경우를 처리하는 코드를 수정 및 위치 변경
    final_df_copy = final_df.copy()
    split_indices = []
    for index, row in final_df.iterrows():
        if len(row['text']) >= 400:
            split_indices.append(index)

    for index in split_indices: # final_df를 순회하면서 텍스트 길이가 400자 이상인 행의 인덱스를 split_indices 리스트에 저장
        row = final_df_copy.loc[index]
        half_length = len(row['text']) // 2 #split_indices에 저장된 각 인덱스에 대해, 해당 행의 텍스트를 반으로 나눔
        part1 = row['text'][:half_length]
        part2 = row['text'][half_length:]
        parts_to_add = [part1, part2]

        final_df_copy.drop(index, inplace=True)
        new_row = row.copy()
  
        for part_to_add in parts_to_add:
            new_row['text'] = part_to_add #원래 긴 텍스트를 포함하고 있던 행은 final_df_copy에서 제거
            #final_df_copy = final_df_copy.append(new_row)  
            final_df_copy = pd.concat([final_df_copy, pd.DataFrame([new_row])], ignore_index=True)     

    final_df_copy['text'] = final_df_copy['text'].apply(lambda x: x.strip())

    # 인덱스 리셋
    final_df_copy.reset_index(drop=True, inplace=True)
    return final_df_copy

#######################################################################################################
### 텍스트 전처리 ###
# split 하는 코드는 이해할 필요 없음.
#######################################################################################################
print("Processing pdf")
chatbot = Chatbot()

pdf_folder = '/data03/hyunaz/project/inner_ft/law_file/mixed' 
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf') and not f.startswith('X')]
df_all = pd.DataFrame(columns=['text', 'CLS1', 'paper_title'])
for p_f in pdf_files:
    pdf_path = os.path.join(pdf_folder, p_f)
    with pdfplumber.open(pdf_path) as pdf:
        paper_text = chatbot.parse_paper(pdf)


    #paper_text = chatbot.parse_paper(full_text)
    #paper_text = chatbot.parse_paper(pdf)
    global df
    df = chatbot.paper_df(paper_text) # split1
    df.columns = ['text'] # 내용 column
    #df['CLS1'] = [p_f.split('_')[1]]*len(df) # 카테고리 column
    #df['paper_title'] = [p_f.split('_')[2].split('[')[0]]*len(df) # 제목 column
    df = split_df(df) # split2
    df_all = pd.concat([df_all, df])
df = df_all.copy()
df.reset_index(drop=True, inplace=True)
#df['text'] = df.apply(lambda x: '[' + x['paper_title'] + '] ' + x['text'].strip(), axis=1) # 내용 앞에 제목 붙임.

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
'''
def generate_paraphrases(prompt, n=3):
    try:
        responses = openai.Completion.create(
            #model="gpt-3.5-turbo-instruct", 
            model="gpt-4-turbo",
            prompt=prompt, 
            max_tokens=1500, 
            n=n, 
            stop=None, 
            temperature=1
        )
        time.sleep(2)
        return [choice['text'].strip() for choice in responses['choices']]
    except Exception as e:
        print(f"Error while generating paraphrases: {e}")
        return []
'''
def generate_paraphrases(prompt, n=2):
    try:
        responses = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],  # 대화형 입력 구조
            max_tokens=1500,
            n=n,
            stop=None,
            temperature=1
        )
        return [choice['message']['content'].strip() for choice in responses['choices']]
    except Exception as e:
        print(f"Error while generating paraphrases: {e}")
        return []

def process_and_save_sentences(sentences, filename):
    data = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_sentence = {executor.submit(generate_paraphrases, f"다음 문장을 다른 방식의 한국어 문장으로 표현해 주세요: '{sentence}'", 2): sentence for sentence in sentences}
        for future in as_completed(future_to_sentence):
            sentence = future_to_sentence[future]
            try:
                paraphrases = future.result()
                for paraphrase in paraphrases:
                    data.append({'original': sentence, 'paraphrase': paraphrase})
            except Exception as e:
                print(f"Failed to process the sentence '{sentence}': {e}")
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
        print(f"Saved paraphrases to {filename}")

# Example usage
sentences = df['text'].tolist()
process_and_save_sentences(sentences, 'Finalparaphrases2.json')



