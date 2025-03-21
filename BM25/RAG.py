import pandas as pd
from rich.markdown import Markdown
from rich.console import Console
import pickle
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from llm import set_seed, set_llm, prompting_answer


# 시드 값 설정
set_seed(42)

def RAG(args):
    
    # retriever 생성 
    year = "2023"
    path = f"index/bm25_{year}.pkl"
    kiwi = pickle.load(open(path, 'rb'))
    kiwi.k = 10 # 10개 문서 검색
    print("\n-- kiwiBM25 retriever load 완료! --")
    print(f'전체 데이터 길이: {len(kiwi.docs)}\n')

    # 쿼리문 설정
    all_texts = ['Mrs Kirkham 치즈가 회수된 이유는 무엇인가?', '코스타리카 내 리스테리아증 감염 사례는 주로 어떤 식품과 관련이 있는가?', '쓰촨 촨라오라오 식품 과학기술유한공사의 식용 식물 혼합유에서 검출된 부적합 물질은 무엇인가?']

    # raw data indexing
    def print_save_texts(docs):
        total_data = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            print(f"[{i+1}] {content}")
            total_data.append(content)
        total_data = '\n'.join(total_data)
        return total_data

    # llm 및 tokenizer 호출
    model_path = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
    # model_path = '/SSL_NAS/concrete/models/models--meta-llama--Meta-Llama-3-8B-Instruct/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e5e23bbe8e749ef0efcf16cad411a7d23bd23298'
    pipeline, tokenizer = set_llm(model_path)
    
    def preprocess_prompt(prompt, max_length=7680):
        tokenized_input = tokenizer(prompt, return_tensors="pt")
        num_tokens = tokenized_input["input_ids"].shape[1] 
        if num_tokens > max_length:
            inputs = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
            return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        return prompt

    # 쿼리 질의 함수 
    def query_answer(question, data, rag = True):
        data = preprocess_prompt(data)
        messages = prompting_answer(question, data, rag)
        output = pipeline(messages, max_new_tokens=512, temperature=0.9, eos_token_id=tokenizer.eos_token_id, do_sample=True)
        # print(output)
        # answer = output[0]["generated_text"][-1]['content'].split('assistant')[0] # llama 쓸 경우
        answer = output[0]["generated_text"][-1]['content'] # EXAONE 쓸 경우
        markdown = Markdown(answer)
        console.print(markdown)
        return answer

    # 응답을 저장할 directory 및 파일 생성 or 불러오기 
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    def make_dir(file_name):
        dir = os.path.join(save_path, f'{file_name}.pkl')
        if os.path.exists(dir):
            file = pickle.load(open(dir, 'rb'))
        else:
            file = {}
        return dir, file
    
    for query_text in all_texts:
        # 유사도 기반 top-k개 search
        print(f"\n질문: {query_text}\n")
        docs = kiwi.search_with_score(query_text)
        total_data = print_save_texts(docs)
        
        rag_path, rag_answer = make_dir('answer_rag')
        data_path, data_dict = make_dir('total_data')
        data_dict[query_text] = total_data
        
        print("\n-- 응답을 생성하는 중입니다 --")
        console = Console()

        # RAG + LLM
        print('\n<RAG 사용 LLM 응답>')
        print(f'- 질문: {query_text}\n')
        answer = query_answer(query_text, total_data, True)
        rag_answer.update({query_text:answer})

        # 응답 저장 
        with open(rag_path, 'wb') as f:
            pickle.dump(rag_answer, f)
            
        with open(data_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    
# config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG')
    args = parser.parse_args()

    # args.data_path = "data/full_list.pkl"
    # args.dataframe_path = "data/full_df.pkl"
    # args.file_path = "data/식품안전정보.xls"
    args.save_path = 'result/250317_exaone'

    RAG(args)