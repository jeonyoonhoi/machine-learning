---
layout: post
title:  "ML Study (5)Dimensionality Reduction"
subtitle:   "2019-02-14-study-machinelearning"
categories: study
tags: machinelearning
comments: true
---

5회차(2/14)** : 2명 (이가경 전은정) (전윤회)

1. Dimensionality Reduction
   - Feature selection, Feature extraction
   - LDA(Linear Discriminant Analysis)
   - PCA(Principal Component Analysis)
   - [Auto Encoder](

# REF

* 각자 자료를 조사하면서 출처 혹은 참고하면 좋을만한 자료를 여기에 남겨주세요

* 유투브 주소)



# Dimensionality Reduction

## What is dimensionality reduction?(차원축소란)

- 주요 변수를 알아나가는 방식으로 random variables 의 수를 줄여나가는 것 
- features을 줄여나가는 과정



#### High-dimensinal Data?

- Biomedical data
- Image data
- Network data
- Text data
- ....



## Why Dimensionality reduction?(차원축소를 왜 하는가?)

- 피쳐수가 많으면 트레이닝 속도가 느리다. 
- 필요 없는 디테일과 노이즈를 필터링 해준다. 
- 데이터 시각화가 편해진다. 



## 특징

- 차원축소가 늘 더 좋은 결과를 내는것은 아니다. 
- 차원 축소가 항 상 단순한 결과를 가져오는 것은 아니다. 
  - 오히려 더 복잡한 결과가 나올 수도 있다.
- Rule of thumb
  - 일단 중복 데이터가 없다는 가정 하에 원래의 데이터로 트레이닝 시킴
  - 그 다음에 차원 축소를 하는 것



## 차원 축소의 종류

- Feature selection : 원래 피쳐의 subset을 선택한다
- Feature extraction : 새로운 피쳐를 만들어내기 위해 피쳐에서 ㅈ



## Feature selection

특징선택은 분류 정확도를 향상시키기 위한 원본 데이터가 주어졌을 때, 가장 좋은 성능을 보여줄 수 있는 데이터의 부분집합을 원본 데이터에서 찾아내는 방법이다. 즉, 분류기의 분류 목적에 가장 밀접하게 연관되어 있는 특징들만을 추출하여 새로운 데이터를 만들어 낸다. 

 이를 통해 데이터의 차원을 줄임으로써 계산 속도를 향상시킬 수 있다. 또한 분류 기준 생성에 방해가 될 수 있는 관계 없는 데이터, 잉여 데이터, 노이즈 들을 제거함으로써 원본 데이터를 그대로 사용 했을 때와 비교하여 성능향상을 기대할 수 있다. 

#### Motivation

- problem 과 가장 관련된 피쳐를 골라내고 관련이 없는 노이즈는 제거하자
- 계산 효율성을 증가시키고, 일반적인 오류를 감소시키기 위함

#### Approach

- 다른 피쳐 서브셋을 점수를 내는 계산을 하는 방식으로 ,새로운 피쳐 서브셋을 만들어 내기 위한 검색 테크닉의 조합으로 이루어진다. 
- 가장 단순한 알고리즘은 오류율을 최소화 하기 위해 가능한 서브셋들을 각각 계산해 보는 것



- Wrapper methods

  - 

- Filter methods

  - 분산이 낮은 것들을 제거 : 
  - Univariate feature selection : X2 테스트

- Embedded mothods

  - 모델 생성 과정의 일환으로 feature selection을 수행한다. 
  - feature selection은 모델을 구체화하고, 그 안에 들어있다.
  - LASSO(Linear regression with L1 regularization )

  

### (1) Filter Method:

- Selt of all Features
- Selecting the best subset
- Learning Algorithm
- Performance

일반적으로 전처리 단계에서 사용되는데, 변수 선택은 어떤 머신러닝 알고리즘과도 독립적이다. 대신 결과 변수와의 상관관계에 근거한 다양한 통계 테스트의 점수로 선택된다. 

- (Feature/Response - 상관계수)
- continuous - continuous - Pearson's Correlation
- continuous - Categorical - LDA
- categorical - continuous - ANOVA
- categorical - categorical - Chi-square



### (2) Wrapper Methods:

![img](https://t1.daumcdn.net/cfile/tistory/9929683359C8CB5E3B)

wrapper method 에서는 변수의 부분집합을 이용하여 mode 를 학습시킨다. 추론을 기반으로 이전 모델로부터 우리는 feature 를 더할건지 뺄건지 결정한다. 문제는 본질적인 검색 문제로 줄어드는데, 이 방방법은 보통 계산 비용이 많이 든다. 

- Forward Selction :  처음 시작은 모델이 아무 변수가 없는 채로 시작한다. 매 반복마다 변수를 추가하여 더 이상 성능 향상이없을 때까지 변수를 추가한다. 
- Backward Elemination : 처음에는 모든 변수를 가지고 시작하지만, 가장 덜 중요한 변수부터 제거하면서 모델의 성능을 향상시킨다. 더 이상 향상이 없을 때까지 반복한다. 
- Recursive Feature Elimination : 이것은 greedy optimization algorithm 인데 가장 좋은 성능을 가진 변수를 찾기 위해서 이다. 모델을 계쏙해섯 ㅐㅇ성하면서 가장 좋거나 나쁜 성능을 내는 모델을 따로 보관하고, 모든 변수가 없어질 때 까지 왼쪽 변수로 다음 모델을 구상한다. 그 다음 제거 순위에 따라 변수의 순위를 매긴다. 



### (3) Embedded Methods :  filter + wrapper

- L1 - based
- Tree-based

filter 와 wrapper 방법의 혼합인데 자체 알고리즘으로 구현된다. 

가장 유면한 방법 중에 LASSO와 RIDGE 가 있는데, 과적합을 줄이기 위해서 내부적으로 penalty 를 주는 방법이다. 

- Lasso :  L1 정규화를 수행하며, 계수 크기의 절댓값과 동일한 penalty를 준다. 
- Ridge : L2 정규화를 수행하는데, 계수 크기의 제곱에 대해 penalty 를 준다. 



--> 계수크기의 절댓값과 동일한 penalty, 계수 크기의 제곱에 대해 penalty 를 주면 왜 과적합이 줄어드나면!

J regularized (세타;X,Y) = J(세타;X,Y)+ 람다 R(세타)

로 나타낼 수 있는데, 

<<J(세타;X,Y)>> 는 훈련집합 X, Y 의 영향을 받지만

 <<람다 R(세타) >> 는 규제항이라  X,Y의 영향을 받지 않는다. R은 단지 가중치의 크기에 제약을 가하는 역할만 하게되고 데이터 집합과 무관하게 원래 있는 사전지식으로 해결가능하다. 

선형회귀에서와의 다른점은 penalty 항이라는 추가 항이 생긴 것

람다 =  ridge function의 alpha 값

alpha값을 변경한다는 것은 penalty term을 제어하게 됨을 뜻함. 

alpha 값이 크면 클수록 penalty 또한 커지게 되면서 계수의 크기가 줄어듭니다. 

이는 변수를 축소하며 다중공선성( multicollinearity) 를 방지하는데 쓰임

*  다중공선성 : 독립변수들 간의 상관관계가 강한경우

* R2 값이 높으면 과최적합

* 해결방법 :  

* ? 상관계수가 높은 독립변수 중 일부를회귀모형에서 제거

  - 나머지 독립변수들의 유의성을 높여주나 필요한 변수의 누락으로 인한 모형설정의 오류 때문에 새로운 문제가 발생할 수 있다. 

  1. 관측값을 늘려 표본 크기를 증가시키낟. 
  2. 원자료에 대해 차분 혹은 로그변환을 하거나 명목변수는 실질변수를 사용한다. 
  3. 사전정보르 이용하여 변수를 상관관계까 높은 다른 변수로 대 체

  

[랏소 릿지](https://brunch.co.kr/@itschloe1/11)

선형<릿지<랏소

각각의 변수별로 coefficient 그래프를 그리면 특정 변수들의 영향력이 특히 높은 것을 볼 수 잇고 이걸 regulation 시켜야 한다. 

다들 비슷비슷하게 만드는게 좋은거라고 생각~~ 

랏소는 0을 만들어 버려서 자동으로 feature selection 도 하는 것

랏소의 경우 alpha 값을 바꾸었을 때 계수가 거의 0에 가까워지지만 작은 알파값에도 계쑤는 이미 완전한 0으로 줄어들었습니다. 따라서 lasso는 중요한 몇 개의 변수만 선택하고 다른 계수들은 0으로 줄입니당 이특징은 feature selection으로 알려져 있고, 릿지의 경우 이 과정이 X . 방법론 적으로 랏소는 절댓값을 더하게 된다. 

EX 10000개의 변수, 서로 correlate

릿지를 쓰면 아직 10000개가 다 살아있어서 복잡

랏소의 문제는 변수들끼리 correlated하다면 단 한개의 변수만 채택하고 다른 변수들 계수를 0으로 바꾸고 정보 손실에의해 정확성이 떨어진다 (극단적이긴 해도)

Elastic Net Regression은 

RidgeLasso 의 하이브리드 형태라고 볼 수 있음





https://brunch.co.kr/@itschloe1/11











### Filter , Wrapper 의 차이점

- Filter 는 종속변수와의 상관관계를 통해 변수의 관련성 측정
- Wrapper 는 실제로 모델을 만들어 변수의 집합의 유용성을 측정
- Filter 는 모델학습이 없기 때문에 속도가 빠르다. 
- Filter는 기능의하위 집합을 평가하는데 통계적 방법 사용
- Wrapper 는 교차 유효성 검사를 사용
- Wrapper 는 Filter 보다 과적합이 발생하기 쉬움



## Feature extraction

많은 상황에서 차원의 크기는 feature의 개수를 의미한다. 문제는 모든 feature 가 전부 의미있는것은 아닐 수도 있다. 주어진 정보들 중에서 정말 의미있는 feature를 뽑아내는 것을 의미한다. 

테그닉으로 나누면 다음과 같다. 

- Projection-based
- Manifold Learning



### PCA( Principal Component Analysis)



### LDA(Linear Discriminant Analysis)



### PCA vs LDA