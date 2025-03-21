from tqdm import tqdm
import numpy as np
import faiss
import torch
import os
import pickle
from sentence_transformers import SentenceTransformer
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# cls 토큰 사용할지
cls_token = False

# embedding path
# embed_path = 'data/embeddings/sroberta'
# embed_path = 'data/embeddings/distiluse'
embed_path = 'data/embeddings/LaBSE'

# sroberta load
def set_bert():
    # model_name = 'jhgan/ko-sroberta-multitask'
    # model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    model_name = 'sentence-transformers/LaBSE'
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    return model

# 텍스트 임베딩 함수 정의
def embed_texts(text_list, model):
    
    embeddings = []        
    # distiluse bert는 output이 (512,)라서 마지막 레이어 거치기 전 값(768,)으로 가져옴
    if cls_token:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        for sentence in tqdm(text_list, desc="Encoding sentences"):
            inputs = model.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            embedding = model[0].auto_model(**inputs).last_hidden_state[:, 0, :].squeeze()
            embeddings.append(embedding.detach().cpu().numpy())
        
    else:
        embeddings = model.encode(text_list, max_length=512, truncation=True, convert_to_numpy=True, show_progress_bar=True)   
        
    return np.array(embeddings, dtype=np.float32)

# index 생성 함수
def make_index(embeddings):
    
    faiss.normalize_L2(embeddings) # 벡터 정규화
    index = faiss.IndexFlatIP(embeddings.shape[1])  # 코사인 유사도 기반 인덱스 생성 (내적 사용)
    index.add(embeddings) # 인덱스에 벡터 추가
        
    return index

def make_shard(df, start, end, model):
    index_shard = faiss.IndexShards(768)
    
    for year in range(start, end+1):
        
        path = f'{embed_path}/embeddings_20{year}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                index_shard.add_shard(make_index(np.array(pickle.load(f), dtype=np.float32)))
        else:
            text_list = df.loc[df['year']==f'20{year}','제목_내용'].to_list()
            index_shard.add_shard(make_index(embed_texts(text_list, model)))
    
    return index_shard