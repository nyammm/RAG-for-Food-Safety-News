import pandas as pd
import faiss
from rich.markdown import Markdown
from rich.console import Console
import pickle
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from index import embed_texts, make_shard, set_bert
from llm import set_seed, set_llm, prompting_answer


# 시드 값 설정
set_seed(42)

def RAG(args):
    
    start = args.start_year
    end = args.end_year
    
    # data load (2016년 9903, 14112행 내용 NaN -> 일단 제거)
    full_df = pd.DataFrame()
    for year in range(start, end+1):
        df = pd.read_excel(args.file_path, sheet_name = f"20{year}", usecols=["제목", "내용"])
        df = df[~df['내용'].isnull()].reset_index(drop=True)
        df["제목_내용"] = df["제목"] + " " + df["내용"]
        df["year"] = f"20{year}"
        full_df = pd.concat([full_df, df]).reset_index(drop=True)

    data = full_df["제목_내용"].to_list()
    
    print("\n-- 데이터 load 완료! --")
    print(f'전체 데이터 길이: {len(data)}\n')

    # index 및 shard 생성
    model = set_bert()
    index_shard = make_shard(full_df, start, end, model)

    # 쿼리문 설정
    query_text = "미국 식품안전검사국이 공중보건경보를 발령한 냉동 닭고기 제품의 제조사는 어디인가?"
    print(f"질문: {query_text}")

    query_embedding = embed_texts([query_text], model)  # 쿼리 임베딩 생성

    print(f"Query Embedding Shape: {query_embedding.shape}")  # (1, 768)인지 확인

    faiss.normalize_L2(query_embedding) # 검색할 쿼리 임베딩도 정규화

    # 유사도 기반 top-k개 search
    distances, indices = index_shard.search(query_embedding, 10) # 병합된 index에서 search

    # raw data indexing
    total_data = [data[i] for i in indices[0]]
    for i, t in enumerate(total_data):
        print(f'({i+1}): {t}')
    total_data = '\n'.join(total_data)

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
    
    rag_path, rag_answer = make_dir('answer_rag')
    # llm_path, llm_answer = make_dir('answer_llm')
    data_path, data_dict = make_dir('total_data')
    data_dict[query_text] = total_data
    
    print("\n-- 응답을 생성하는 중입니다 --\n")
    console = Console()

    # only LLM
    # print('<RAG 미사용 LLM 응답>')
    # print(f'- 질문: {query_text}\n')
    # answer = query_answer(query_text, total_data, False)
    # llm_answer.update({query_text:answer})

    # RAG + LLM
    print('\n<RAG 사용 LLM 응답>')
    print(f'- 질문: {query_text}\n')
    answer = query_answer(query_text, total_data, True)
    rag_answer.update({query_text:answer})

    # 응답 저장 
    with open(rag_path, 'wb') as f:
        pickle.dump(rag_answer, f)
        
    # with open(llm_path, 'wb') as f:
    #     pickle.dump(llm_answer, f)
        
    with open(data_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    
# config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAG')
    args = parser.parse_args()

    args.data_path = "data/full_list.pkl"
    args.dataframe_path = "data/full_df.pkl"
    args.file_path = "data/식품안전정보.xls"
    args.start_year = 23
    args.end_year = 23
    args.save_path = 'result/250304_exaone'

    RAG(args)