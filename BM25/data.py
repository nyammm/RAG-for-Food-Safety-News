import pandas as pd
from langchain_teddynote.retrievers import KiwiBM25Retriever
import pickle
from tqdm import tqdm


path = "data/식품안전정보.xls"

for year in tqdm(range(14, 25), desc="BM25 인덱스 생성 중"):
    df = pd.read_excel(path, sheet_name = f"20{year}", usecols=["제목", "내용"])
    df = df[~df['내용'].isnull()].reset_index(drop=True)
    df["제목_내용"] = df["제목"] + " " + df["내용"]
    texts = df['제목_내용'].to_list()
    print(f"[{year}]",len(texts))
    kiwi = KiwiBM25Retriever.from_texts(texts)
    with open(f'index/bm25_20{year}.pkl', 'wb') as f:
        pickle.dump(kiwi, f)