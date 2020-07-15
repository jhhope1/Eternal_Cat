
import json
import re
from collections import Counter
from typing import *

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd
from khaiii import KhaiiiApi  # khaiii 레포는 https://github.com/kakao/khaiii 이쪽


def re_sub(series: pd.Series) -> pd.Series:
    series = series.str.replace(pat=r'[ㄱ-ㅎ]', repl=r'', regex=True)  # ㅋ 제거용
    series = series.str.replace(pat=r'[^\w\s]', repl=r'', regex=True)  # 특수문자 제거
    series = series.str.replace(pat=r'[ ]{2,}', repl=r' ', regex=True)  # 공백 제거
    series = series.str.replace(pat=r'[\u3000]+', repl=r'', regex=True)  # u3000 제거
    return series

def flatten(list_of_list : List) -> List:
    flatten = [j for i in list_of_list for j in i]
    return flatten

def get_token(title: str, tokenizer)-> List[Tuple]:
    
    if len(title)== 0 or title== ' ':  # 제목이 공백인 경우 tokenizer에러 발생
        return []
    
    result = tokenizer.analyze(title)
    result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]  # (형태소, 품사) 튜플의 리스트
    return result

def get_all_tags(df) -> List:
    tag_list = df['tags'].values.tolist()
    tag_list = flatten(tag_list)
    return tag_list

train['plylst_title'] = re_sub(train['plylst_title'])
train.loc[:, 'ply_token'] = train['plylst_title'].map(lambda x: get_token(x, tokenizer))
using_pos = ['NNG','SL','NNP','MAG','SN']  # 일반 명사, 외국어, 고유 명사, 일반 부사, 숫자
train['ply_token'] = train['ply_token'].map(lambda x: list(filter(lambda x: x[1] in using_pos, x)))
unique_tag = set(token_itself)
unique_word = [x[0] for x in unique_tag]
# 우리의 목적은 정답 tags를 맞추는 것이기 때문에 정답 tags에 나온 형태소만 남겨둡니다.
train['ply_token'] = train['ply_token'].map(lambda x: list(filter(lambda x: x[0] in unique_word, x)))
train.head(10)
