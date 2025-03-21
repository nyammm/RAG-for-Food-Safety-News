from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm


# 연도
year = 2014

# 데이터 다운로드 및 편집
data = pd.read_excel('data/식품안전정보.xls', sheet_name = str(year), usecols=["제목", "내용"])
data = data[~data['내용'].isnull()].reset_index(drop=True)
data["제목_내용"] = data["제목"] + " " + data["내용"]
data = data['제목_내용'].to_list()

# 임베딩 저장 경로 설정
embedding_path = f'data/embeddings/LaBSE/embeddings_{year}.pkl'
embeddings = []
    
print('임베딩 추출할 데이터 길이:',len(data))

# loading model
# model_path = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
# model_path = 'jhgan/ko-sroberta-multitask'
model_path = 'sentence-transformers/LaBSE'
model = SentenceTransformer(model_path)
model.max_seq_length = 512

embeddings = model.encode(data, max_length=512, truncation=True, convert_to_numpy=True, show_progress_bar=True)
    
with open(embedding_path, 'wb') as f:
    pickle.dump(embeddings, f)
        
print("임베딩 추출 완료!")