from transformers import pipeline
import leidenalg as la  # Leiden 알고리즘
from llm import set_seed, set_llm
set_seed(2025)
from rich.markdown import Markdown
from rich.console import Console
import pickle
import os


# 날짜
date = "250305"

# 모델 불러오기
model_path = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
pipeline, tokenizer = set_llm(model_path)

# 2023년도 데이터 
document = "홍콩 식품안전센터, 일본산 채소 제품 '식품안전명령' 위반 의심 식품환경위생서 식품안전센터는 금일(9월 15일), 일본산 식품 검사 중 규제 지역에서 수입된 채소를 발견했다며, 해당 수입업체가 '식품안전명령(이하 '명령')'을 위반한 것으로 의심된다고 발표함. 센터는 해당 제품을 이미 봉인해 시중에 유통되지 않았으며, 사건을 조사 중임. 센터는 해당 일본산 식품의 식품 라벨을 검사하던 중 도치기현 버섯(菇類) 식품에 방사능증명서 및 수출업체증명서가 첨부되지 않은 것을 발견했으며, 이는 '명령'을 위반한 것으로 의심됨. '명령'에 따르면, 후쿠시마현의 모든 채소, 과일, 유(奶), 유음료 및 분유 수입이 금지됨. 지바, 도치기, 이바라키, 군마에서 이러한 식품을 수입하는 경우에는 일본 주무 당국에서 발급한 방사능증명서 및 수출업체증명서를 반드시 첨부하여 해당 식품의 방사능 수준이 지침 제한치를 초과하지 않으며 섭취에 적합함을 증명해야만 수입될 수 있음.  센터는 계속해서 사건을 추적하고 일본 관계당국에 사건 통지 등을 포함한 적절한 조치를 취할 것임. 증거가 충분할 경우, 해당 수입업체를 기소할 것임."

# 요소 추출 프롬프트
instance_extract_prompt = '''
-목표-
텍스트 문서와 엔터티 유형 목록이 주어졌을 때, 텍스트에서 엔터티 유형 목록에 해당되는 모든 엔터티를 식별하고, 식별된 엔터티 간의 모든 관계를 찾아내시오.

**⚠ 중요: 아래 명시된 엔터티 유형만 사용할 수 있으며, 이 목록에 없는 유형은 절대 생성하지 마시오.**
- **허용된 엔터티 유형 목록**:
  - **FOOD**: 특정 식품 또는 식품 관련 제품 (예: 치즈, 고추 스틱, 코코넛음료)
  - **DISEASE**: 전염병 및 공중보건 관련 질병 (예: 조류 인플루엔자, 코로나19, 식중독, 아프리카돼지열병)
  - **HARMFUL_MICROORGANISM**: 식품 속 유해 미생물 (예: 살모넬라, 리스테리아, 대장균, 노로바이러스)
  - **HAZARDOUS_SUBSTANCE**: 식품 속 유해 화학물질 및 방사성 물질 (예: 방사성 세슘, 유리 조각, 중금속, 미세 플라스틱)
  - **GOVERNMENT_AGENCY**: 정부 및 공공 기관 (예: 식품의약품안전처, 미국 FDA, 후생노동성, 공화당, 의회)
  - **ORGANIZATION**: 국제 기구 및 다국적 협의체 (예: WHO, UN, WTO, EU, 식품안전청)
  - **LAW**: 식품 및 보건 관련 법규 및 제도 (예: HACCP, 식품위생법, CODEX)
  - **COMPANY**: 식품 제조·유통·관련 기업 (예: CJ제일제당, 롯데푸드, 스타벅스)
  - **RESEARCH_INSTITUTE**: 식품 및 건강 관련 연구 기관 (예: 국립식량과학원, 한국식품연구원)
  - **LOCATION**: 장소 및 지리적 정보 (예: 특정 식료품점, 특정 마트)
  - **COUNTRY**: 국가 (예: 대한민국, 미국, 일본)
  - **PERSON**: 공무원, 연구자, 전문가 등 (예: 식약처 대변인, 식품 안전 연구원)

-단계-
1. 모든 엔터티를 식별합니다. 엔터티는 사람, 사물, 개념 등이 될 수 있으며, 어떠한 행동이나 관계는 엔터티가 될 수 없습니다. 각 엔터티에 대해 다음 정보를 추출합니다:
   - entity_name: 엔터티의 이름
   - entity_type: 허용된 엔터티 유형 목록 중 하나
   - entity_description: 엔터티의 속성과 활동을 포괄적으로 설명한 내용  
   각 엔터티의 형식:  
   ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 1단계에서 식별된 엔터티들 중에서 *명확하게 관련이 있는* (source_entity, target_entity) 쌍을 찾아냅니다.  
   각 엔터티 쌍에 대해 다음 정보를 추출합니다:
   - source_entity: 출처 엔터티 이름 (1단계에서 식별한 엔터티 중 하나)
   - target_entity: 대상 엔터티 이름 (1단계에서 식별한 엔터티 중 하나)
   - relationship_description: 출처 엔터티와 대상 엔터티가 서로 관련이 있다고 판단한 이유
   - relationship_strength: 출처 엔터티와 대상 엔터티 간의 관계 강도를 나타내는 숫자 점수  
   각 관계의 형식:  
   ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. 위 1, 2단계에서 식별한 모든 엔터티와 관계를 {record_delimiter}를 목록 구분자로 사용하여 각각 하나의 리스트로 출력합니다.
   entity 리스트 형식: 
   [<entity_tuple_1>, <entity_tuple_2>, ..., <entity_tuple_n>]
   relationship 리스트 형식: 
   [<relationship_tuple_1>, <relationship_tuple_2>, ..., <relationship_tuple_n>]

######################
-예제-
######################

예제 1:
텍스트:  
"일본 후생노동성, 식품 중 방사성물질 조사결과(2023년 2~3월 조사분) 일본 후생노동성은 28일, 식품 중 방사성물질 조사 결과(2023년 2~3월 조사분)을 발표함. 방사선량은 기준치 설정 근거인 연간 선량 1 mSv 의 0.1% 정도로 밝혀짐. 상세 내용은 다음과 같음.  -후생노동성은 국립의약품 식품위생연구소에 위탁하여 2023년 2월부터 3월에 일본 전국 15지역에서 실제로 유통되는 식품을 구입하여 식품 중 방사성 세슘으로부터 받는 연간 방사선량을 추정하였음.-조사 결과, 식품 중 방사성 세슘으로부터 사람이 1년간 받는 방사선량은 0.0005 ~ 0.0010 mSv/y 로 추정되며, 이는 현행 기준치의 설정 근거인 연간 상한 선량 1 mSv/y 의 0.1% 정도로 극히 적다는 것을 확인할 수 있었음. -또한 방사성 세슘(Cs-134 와 Cs-137 의 합계) 농도가 0.5 Bq/kg 이상이 된 시료에 대해서는 방사성 스트론튬(Sr-90) 및 플루토늄(Pu-239 + 240)도 조사하도록 하고 있음.-이번에, 조사 대상이 되는 방사성 세슘 농도가 0.5 Bq/kg 이상인 시료는 없었음.-후생노동성에서는 앞으로도 계속적으로 동일한 조사를 실시하여 식품의 안전성 검증 노력을 할 것임.  참고: 도쿄전력 후쿠시마 제1원자력발전소 사고에서 유래하여, 식품 중 방사성 물질로부터 장기적으로 입는 선량의 대부분은 방사선 세슘에 의한 것이라고 알려져 있음."

출력:
entity_list = 
[("entity", "일본 후생노동성", "GOVERNMENT_AGENCY", "일본의 보건 및 식품 안전을 담당하는 정부 기관"),
("entity", "국립의약품 식품위생연구소", "RESEARCH_INSTITUTE", "일본의 식품 및 의약품 안전 연구를 수행하는 연구소"),
("entity", "방사성 세슘", "HAZARDOUS_SUBSTANCE", "식품에서 검출될 수 있는 방사성 물질, 주로 후쿠시마 원전 사고 이후 문제가 됨"),
("entity", "연간 선량 1 mSv", "LAW", "식품 방사성 물질 기준을 정하는 법적 기준"),
("entity", "도쿄전력 후쿠시마 제1원자력발전소", "LOCATION", "2011년 사고로 방사성 물질 유출이 발생한 원전")]

relationship_list = 
[("relationship", "일본 후생노동성", "국립의약품 식품위생연구소", "후생노동성이 연구소에 식품 방사성 물질 조사를 위탁함", 9),
("relationship", "방사성 세슘", "연간 선량 1 mSv", "방사성 세슘의 방사선량이 법적 기준치의 0.1% 수준임", 8),
("relationship", "방사성 스트론튬", "방사성 세슘", "방사성 세슘 농도가 높으면 방사성 스트론튬도 검출 가능성이 있음", 7),
("relationship", "도쿄전력 후쿠시마 제1원자력발전소", "방사성 세슘", "후쿠시마 원전 사고로 인해 방사성 세슘이 주요 오염원으로 보고됨", 10),
("relationship", "일본 후생노동성", "방사성 세슘", "후생노동성이 식품 중 방사성 세슘 오염도를 평가함", 9),
("relationship", "일본 후생노동성", "일본 전국 15지역", "일본 후생노동성이 15개 지역에서 방사성 물질 조사를 수행함", 8)]

######################

예제 2:
텍스트:  
"베트남 보건부, 식품안전 관련 행정위반 업체 처벌  결정  베트남 보건부는 9월 15일, 다음의 행정위반 업체에 대한 처벌을 결정했음(Số: 23/QĐ-XPHC) -업체명:  MTV Kinh doanh thương mại Việt Mỹ 유한책임회사(위치: 남딘성)>위반행위: 보관창고 규정 없음, 식품안전교육 미수료 *자세한 내용은 원문참조**원문링크:https://emohbackup.moh.gov.vn/publish/home?documentId=9232"

출력:
entity_list = 
[("entity", "베트남 보건부", "GOVERNMENT_AGENCY", "베트남의 보건 및 식품 안전을 담당하는 정부 기관"),
("entity", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "COMPANY", "베트남 남딘성에 위치한 식품 관련 사업을 운영하는 회사"),
("entity", "남딘성", "LOCATION", "베트남의 행정 구역 중 하나"),
("entity", "식품안전교육", "LAW", "식품 안전 관련 교육 및 규정을 준수해야 하는 제도"),
("entity", "보관창고 규정", "LAW", "식품 보관을 위한 위생 및 안전 기준을 정하는 규정")]

relationship_list = 
[("relationship", "베트남 보건부", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "베트남 보건부가 해당 기업의 행정 위반을 적발하고 처벌을 결정함", 9),
("relationship", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "남딘성", "해당 기업이 남딘성에 위치함", 7),
("relationship", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "식품안전교육", "해당 기업이 식품안전교육을 이수하지 않음", 8),
("relationship", "MTV Kinh doanh thương mại Việt Mỹ 유한책임회사", "보관창고 규정", "해당 기업이 보관창고 관련 규정을 준수하지 않음", 8)]

######################
-실제 데이터-
######################
텍스트: "{input_text}"  
######################  
출력:
'''
instance_extract_prompt = instance_extract_prompt.replace('{input_text}', document)
instance_extract_prompt = instance_extract_prompt.replace('{tuple_delimiter}', ', ')
instance_extract_prompt = instance_extract_prompt.replace('{record_delimiter}', ', ')

# 응답 생성 
print("\n-- 응답을 생성하는 중입니다 --\n")
console = Console()
output = pipeline(instance_extract_prompt, max_new_tokens=768, temperature=0.9, eos_token_id=tokenizer.eos_token_id, do_sample=True)
answer = output[0]['generated_text'].split('출력:')[-1]

markdown = Markdown(answer)
console.print(markdown)

path = 'graphRAG/result'
try:
    instances = pickle.load(open(f'{path}/{date}/instances.pkl','rb'))
except:
    os.makedirs(f'{path}/{date}', exist_ok=True)
    instances = {}

instances[document] = answer

with open(f'{path}/{date}/instances.pkl','wb') as f:
    pickle.dump(instances, f)