import os
from llm import set_seed, set_llm
set_seed(2025)
import pickle
from rich.markdown import Markdown
from rich.console import Console
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import pipeline


# 날짜
date = "250306"

# 모델 불러오기
model_path = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
pipeline, tokenizer = set_llm(model_path)

# 데이터
document = "홍콩 식품안전센터, 일본산 채소 제품 '식품안전명령' 위반 의심 식품환경위생서 식품안전센터는 금일(9월 15일), 일본산 식품 검사 중 규제 지역에서 수입된 채소를 발견했다며, 해당 수입업체가 '식품안전명령(이하 '명령')'을 위반한 것으로 의심된다고 발표함. 센터는 해당 제품을 이미 봉인해 시중에 유통되지 않았으며, 사건을 조사 중임. 센터는 해당 일본산 식품의 식품 라벨을 검사하던 중 도치기현 버섯(菇類) 식품에 방사능증명서 및 수출업체증명서가 첨부되지 않은 것을 발견했으며, 이는 '명령'을 위반한 것으로 의심됨. '명령'에 따르면, 후쿠시마현의 모든 채소, 과일, 유(奶), 유음료 및 분유 수입이 금지됨. 지바, 도치기, 이바라키, 군마에서 이러한 식품을 수입하는 경우에는 일본 주무 당국에서 발급한 방사능증명서 및 수출업체증명서를 반드시 첨부하여 해당 식품의 방사능 수준이 지침 제한치를 초과하지 않으며 섭취에 적합함을 증명해야만 수입될 수 있음.  센터는 계속해서 사건을 추적하고 일본 관계당국에 사건 통지 등을 포함한 적절한 조치를 취할 것임. 증거가 충분할 경우, 해당 수입업체를 기소할 것임."
entity_description = "홍콩 식품안전센터, 일본산 채소 제품, 도치기현 버섯, 후쿠시마현, 식품안전명령, 지바현, 이바라키현, 군마현, 홍콩 관계당국, 일본 주무 당국"

# claim 추출 prompt
claim_extract_prompt = '''
-목표 활동-
당신은 특정 entity에 대한 주장(claim)을 분석하는 인간 분석가를 돕는 지능형 어시스턴트입니다.

-목적-
주어진 텍스트 문서와 미리 정의된 entity 목록, claim 유형 목록이 주어졌을 때, 해당 claim 유형과 일치하며 entity 목록과 관련된 모든 claim을 식별하세요.

-단계-

1. 텍스트에서 **미리 정의된 entity 목록에 포함된 entity가 존재하는지 확인**합니다.  

2. 텍스트에서 아래 claim 유형 목록 중 해당되는 claim 유형을 찾습니다.
   - claim 유형 목록:
    - **FOOD RECALL**: 식품에서 유해 물질(세균, 알레르기 유발 물질, 화학 물질 등)이 검출되어 리콜 조치됨  
      (예: "[회수] 영국 Chiltern Artisan, 대장균 오염으로 고추 스틱 회수")
    - **FOOD CONTAMINATION**: 식품에서 미생물(대장균, 살모넬라, 리스테리아 등) 또는 유해 물질이 검출됨  
      (예: "[식중독] 영국 스코틀랜드, 병원성 대장균으로 인한 치즈 회수 이후 1명 사망")
    - **ALLERGEN MISLABELING**: 식품에 알레르기 유발 성분(우유, 땅콩 등)이 포함되었으나 표시가 누락됨  
      (예: "[회수] 아일랜드 식품안전청, 알레르기 유발물질 우유 함유 미표시로 코코넛음료 회수 공지")
    - **ILLEGAL FOOD ACTIVITY**: 불법 제조, 유통, 허위 표시, 식품 보관 위반, 식품 사기 등의 법 위반 행위  
      (예: "가짜 양고기 판매 혐의로 중닝 해산물가게 적발")
    - **REGULATORY ACTION**: 정부 기관이 내린 행정 조치 (판매 금지, 작업 중단, 벌금 등)  
      (예: "싱가포르 법원, 미허가 냉장 창고 불법 운영 혐의로 1개 업체 및 대표 벌금형 선고")
    - **FOOD SAFETY ADVISORY**: 식품 관련 보건 당국의 공식 발표, 소비자 주의 경보  
      (예: "싱가포르 식품청, 식품용 스티로폼 용기는 적절하게 사용할 경우 안전해")

3. 식별된 claim에 대해 다음 정보를 추출하세요:  
   - **subject_entity**: claim의 주체가 되는 entity의 이름(주어진 entity 목록 중 하나)
   - **object_entity**: claim의 영향을 받는 entity(주어진 entity 목록 중 하나, 존재하지 않는 경우 **NONE** 반환)
   - **claim_type**: claim의 유형(미리 정의된 claim 유형 중 하나) 
   - **claim_status**: **TRUE**, **FALSE**, 또는 **SUSPECTED**
     - TRUE: claim이 확인된 경우  
     - FALSE: claim이 거짓으로 판명된 경우  
     - SUSPECTED: claim이 검증되지 않은 경우  
   - **claim_description**: claim의 상세 설명(관련 증거 및 참고 자료 포함)
   - **claim_date**: claim이 제기된 날짜(start_date, end_date)
     - 단일 날짜의 경우 시작일과 종료일을 동일하게 설정
     - 날짜를 알 수 없는 경우 **NONE** 반환
   - **claim_source**: claim과 관련된 원문에서의 **모든 관련 인용문 목록**

   각 claim을 다음 형식으로 출력하세요:
    (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

4. 1,2,3단계에서 식별된 모든 claim을 단일 리스트로 출력하세요. 리스트 항목 간 구분자로 {record_delimiter}를 사용하세요.

######################
-예제-
######################

예제 1:
entity 목록: "신베이시 위생국, 중닝 해산물가게, 난타이궈지 기업사, 양씨 남매, 돼지고기, 양고기, 식품안전위생관리법, 가짜 양고기, 대만, 다타이페이 과채시장"
텍스트: "대만 신베이시, 돼지고기를 양고기로 둔갑...관련 업체에 작업 중단 및 판매 금지 명령 신베이시(新北) 위생국은 금일(25일), 검찰과 합동으로 가짜 양고기를 판매한 것으로 의심되는 다타이페이 과채시장(DaTaiPei:大台北果菜市場)의 '중닝 해산물가게(ZhongNing:重寧海產號)', 육류 공급업체 '난타이궈지 기업사(NanTaiGuoJi:南台國際企業社)'를 적발했음. 혼입 또는 위조 혐의로 식품안전위생관리법에 의거, 상기 2개 업체에 작업 잠정 중단 및 판매 중지를 명령함. 검경 수사에 따르면, 2020년부터 공급업체의 천(陳)씨는 '중닝 해산물가게' 양(楊)모 남매의 주문을 받고, 돼지고기가 혼합된 양고기 또는 양고기가 함유되지 않은 돼지고기 슬라이스를 양씨 남매에게 판매했으며, 양씨 남매는 이를 양고기로 둔갑시켜 소비자에게 판매한 것으로 확인됨."
출력: [("중닝 해산물가게", "NONE", "ILLEGAL FOOD ACTIVITY", "TRUE", "2024-02-25T00:00:00", "2024-02-25T00:00:00", "중닝 해산물가게는 가짜 양고기를 판매한 혐의로 적발되었다.", "신베이시(新北) 위생국은 금일(25일), 검찰과 합동으로 가짜 양고기를 판매한 것으로 의심되는 다타이페이 과채시장(DaTaiPei:大台北果菜市場)의 '중닝 해산물가게(ZhongNing:重寧海產號)'를 적발했음."), 
("난타이궈지 기업사", "양씨 남매", "ILLEGAL FOOD ACTIVITY", "TRUE", "2020-01-01T00:00:00", "2024-02-25T00:00:00", "난타이궈지 기업사는 2020년부터 양씨 남매에게 돼지고기가 혼합된 양고기를 공급한 혐의가 있다.", "검경 수사에 따르면, 2020년부터 공급업체의 천(陳)씨는 '중닝 해산물가게' 양(楊)모 남매의 주문을 받고, 돼지고기가 혼합된 양고기 또는 양고기가 함유되지 않은 돼지고기 슬라이스를 양씨 남매에게 판매했음."), 
("신베이시 위생국", "중닝 해산물가게", "REGULATORY ACTION", "TRUE", "2024-02-25T00:00:00", "2024-02-25T00:00:00", "신베이시 위생국은 중닝 해산물가게에 작업 중단 및 판매 금지 명령을 내렸다.", "신베이시(新北) 위생국은 혼입 또는 위조 혐의로 식품안전위생관리법에 의거, 상기 2개 업체에 작업 잠정 중단 및 판매 중지를 명령함.")]

예제 2:
entity 목록: "베트남 보건부, MTV Kinh doanh thương mại Việt Mỹ 유한책임회사, 남딘성, 식품안전교육, 보관창고 규정"
텍스트: "베트남 보건부, 식품안전 관련 행정위반 업체 처벌  결정  베트남 보건부는 9월 15일, 다음의 행정위반 업체에 대한 처벌을 결정했음(Số: 23/QĐ-XPHC) -업체명:  MTV Kinh doanh thương mại Việt Mỹ 유한책임회사(위치: 남딘성)>위반행위: 보관창고 규정 없음, 식품안전교육 미수료 *자세한 내용은 원문참조**원문링크:https://emohbackup.moh.gov.vn/publish/home?documentId=9232"
출력: [("베트남 보건부", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "REGULATORY ACTION", "TRUE", "2023-09-15T00:00:00", "2023-09-15T00:00:00", "베트남 보건부는 MTV Kinh doanh thương mại Việt Mỹ 유한책임회사가 식품안전 관련 규정을 위반했다고 판단하여 행정 처벌을 결정하였다.", "베트남 보건부는 9월 15일, 다음의 행정위반 업체에 대한 처벌을 결정했음(Số: 23/QĐ-XPHC)"),
("MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "보관창고 규정", "ILLEGAL FOOD ACTIVITY", "TRUE", "2023-09-15T00:00:00", "2023-09-15T00:00:00", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사는 보관창고 규정을 준수하지 않아 행정 처벌을 받았다.", "위반행위: 보관창고 규정 없음"), 
("MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "식품안전교육", "ILLEGAL FOOD ACTIVITY", "TRUE", "2023-09-15T00:00:00", "2023-09-15T00:00:00", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사는 식품안전교육을 이수하지 않아 규정을 위반하였다.", "위반행위: 식품안전교육 미수료")]

######################
-실제 데이터-
######################
entity 목록: "{entity_description}"
텍스트: "{input_text}"  
######################  
출력: 
'''
claim_extract_prompt = claim_extract_prompt.replace('{input_text}', document)
claim_extract_prompt = claim_extract_prompt.replace('{entity_description}', entity_description)
claim_extract_prompt = claim_extract_prompt.replace('{tuple_delimiter}', ', ')
claim_extract_prompt = claim_extract_prompt.replace('{record_delimiter}', ', ')

# 응답 생성 
print("\n-- 응답을 생성하는 중입니다 --\n")
console = Console()
output = pipeline(claim_extract_prompt, max_new_tokens=768, temperature=0.9, eos_token_id=tokenizer.eos_token_id, do_sample=True)
answer = output[0]['generated_text'].split('출력:')[-1]

markdown = Markdown(answer)
console.print(markdown)

path = 'graphRAG/result'
try:
    instances = pickle.load(open(f'{path}/{date}/claims.pkl','rb'))
except:
    os.makedirs(f'{path}/{date}', exist_ok=True)
    instances = {}

instances[document] = answer

with open(f'{path}/{date}/claims.pkl','wb') as f:
    pickle.dump(instances, f)