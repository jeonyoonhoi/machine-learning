---
layout: post
title:  "ML Study (1)Introduction "
subtitle:   "2019-01-17-study-machinelearning"
categories: study
tags: machinelearning
comments: true
---



## Machine Learning Applications

머신러닝을 공부해서 어디에 사용할까요?

- Speech raecognition 
- Fraud detection 
- Recommendation
- Face recognition
- Email filtering
- Medical diagnosis
- Weather prediction



## Machine Learning이란 ? 

#### 개념

머신러닝이란 컴퓨터가 데이터를 통해 학습할 수 있는 능력을 주는 연구분야를 말합니다. 기존에는 데이터를 통해 사람이 학습을 하고, 컴퓨터에게 이를 지시하는 방식(with being explicitly programmed)으로 이루어 지던 과정에서 사람이 이해하고 컴퓨터에게 입력하는 단계가 빠진 것입니다. 



#### Tom Mitchell, 1977

A computer program is said to learn from <u>experience E</u> with respect to some <u>task T</u> and some performance measure P, if its performance on T, as measured by P improves with experience E. 

예시로 스팸메일을 필터링 하는 특정 컴퓨터 프로그램을 들겠습니다. 이 프로그램은 경험을 통해 학습합니다. Experience, 여기서의 경험은 과거의 스팸메일과 일반적인 메일들의 예시, 즉 학습에 사용되는 데이터라고도 볼 수 있습니다. 이 경험은 특정 Task, 스    팸 메일을 예측에 대한 것입니다. 또한 스팸메일을 얼마나 정확하게 분류해내는가에 대한 정확도 혹은 성능을 나타내는 것이 Perfomance measure P 입니다. 작업에 대한 성능은 경험으로 인해 향상됩니다.  이때 사람이 할 일은 프로그램, 머신러닝 알고리즘을 어떻게 만들 것인지를 고민해야 합니다. 



#### 모델 설계

실제 세계가 존재하고 이것을 추상화 하여 어떤 수학적인 프로그래밍 결과를 만들어 내는 일입니다. 

1. 구성요소 decision value(x)

2. 관계 constraint( 제약조건 ) 

3. 목적, 목표 objective function (목적함수)

   costfunction, 목적함수에  decision value(x,y) 를 넣어서  선형을 띄는(제약조건) 모델을 알고리즘을 통해 최적화를 시키게 됩니다. 



#### 모델을 만들기 위한 가이드라인

1. 모델의 목적은 무엇인가? (예측, 추천, 인식, 등 어떤 모델인가)
2. 정확도를 얼마나 요구하는가?
3. 이 모델/ 결과물을 누가 사용할 것인가?
4. 자원의 제한(시간, time, data 등)

## Why use Machine Learning?

왜 머신러닝을 사용할까요? 너무다 당연한 대답이지만 이전보다 사람이 해야 할 일이 훨씬 줄었기 때문이죠! 

기존의 방식대로라면 사람들이 직접 스팸메일을 보고 특징을 분석하고 패턴을 추출하여 스팸메일을 감지하는 알고리즘을 짜야 합니다. 이 과정을 반복하는 테스트를 통해 성능을 높입니다. 약점은 직접 패턴을 알아내야 하고, 프로그램이 그저 복잡한 규칙의 리스트가 되어버려 유지가 힘들어진다는 것입니다. 

머신러닝 기반의 접근을 한다면 스팸 메일, 일반메일 데이터 셋이 컴퓨터에서 주어졌을 때 스팸메일만의 패턴을 컴퓨터가 스스로 찾게 됩니다. 강점은 프로그램이 훨씬 더 짧아지고 정확해 진다는 점과, 사람의 개입 없이도 스스로 변화에 적응하고 새로운 패턴을 찾아낼 수 있다는 것입니다. 

머신러닝이 무조건 좋다고는 할 수 없지만 다음과 같은 특징을 가진 문제들에 있어서는 강점을 발휘할 수 있습니다. 

- 전통적인 방법으로 접근하기에 너무 복잡한 문제들
- 기존 알려진 알고리즘이 없는 문제(새로운 답이 필요)



## ML vs Data Mining

#### Data Mining

- Data mining is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statisitcs, and database systems. 

  데이터 마이닝이란 머신러닝, 통게학, 데이터베이스 시스템의 교점의 방법들을 통해 큰 데이터셋에서 패턴을 추출하는과정을 말합니다. 

  예를 들어 아이폰을 사용하는 사람들의 특징이 무엇인지 추출하고 싶을 때 나이, 학력, 수입, 등을 통계학적 방법으로 추출하고, 이 사람들을 특징에 따라 그룹화 하고 싶을 때 머신러닝 방법인 클러스터링을 사용할 수 있습니다. 

## ML vs Data Science

#### Data Science

- Data Science is the study of the generalizable extraction of knowledge from data

- Data Science is an emerging area of work concerned with the collection, preparation, analysis visualization, management, and preservation of largle collections of information 

  데이터로부터 일반화될수 있는 지식의 추출을연구하는 것. 수집, 전처리, 분석 시각화, 관리, 그리고 큰 규모의 정보 보유에 관심을 갖는 영역. 



- Data Science is an academc program and they teach
  - Data mining
  - Machine Learning
  - Natural Language PRocessing
  - Information Retrieval



## ML vs Artificial Intelligence

#### Artificial Intelligence

- Artifical intelligence , sometimes called machine intelligence, is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and other animals.

- Artificial intelligence is used when a machine mimics "cognitive" function, such as "learning" and  "problem solving"

  머신러닝은 AI의 앞 과정이라고 할 수 있습니다. 즉 스스로 학습한 기계가 사람의 학습이나 문제해결과정 같은 인지적 기능을 수행하는 것이 AI 입니다. 사람이나 동물의 지능처럼 기계가 어떤 현상을 이해하고, 상황에 대한 판단과 의사결정 및 합리적 추론을 하는 행위를 수행하는 것. 

- Aritificial intelligence includes /will perform

  - Knowlege reasoning
  - Robotics
  - Autonomous driving
  - Machine translation

  

## ML vs Deep Learning



##  Type of Machine Learning

* 훈련동안 주어지는 데이터의 양과 종류에 따라 

1. Supervised learning (지도학습)

   지도학습은 학습시킬 때 labeling 된 데이터를 주고 하는 학습이에요 문제와 답을 알려주고 새로운 문제에 대해 답을 출력할 수 있는 모델을 만들게 됩니다. 

   - Classification & Regression 

     분류와 회귀 사실 같은 원리인데 classification 은 이미 정해져 있는 0,1 / T,F / A,B,C,D 등 정해진 라벨을 알아내는 것이고, regression은 특정 numeric value를 알아내는 것입니다.

     

2. Unsupervised learning (비지도학습)

   답을 알려주지도 않고 답을 맞추게 한다는 것이 뭔가 와닿지 않을 수도 있지만 그래서 실제 세상에서는 더욱 중요하다고 느끼는 방법입니다. 모든 데이터에 라벨링이 되어있다면 참 좋겠지만 그런 정제된 데이터는 지극히 한정적이니까요. 

   비지도 학습은 보통 어떤 데이터를 분석하기 전에 EDA 과정(Explore Dataset Anlysis) 에서 많이 사용하게 됩니다. 데이터를 좀 더 잘 이해하기 위함이죠. 

   - Clustering & Dimensionality Reduction 

     예시로 사람을 들자면 작은 집단에서 특징이 두개 주어졌다면 이것들을 분류하거나 집단을 세분화 하는 일이 어렵지 않을 것입니다. 그런데 사람의 수와 특징의 수가 커져서 데이터셋이 한눈에 들어오지 않는다면 라벨링을 하는 방법에 대해 고민이 필요하겠죠. 

     고객들의 집합을 세분화하여 특정 타겟을 대상으로 마케팅을 하고 싶을 때 사람들의 특징에 대한 데이터에 대해 클러스터링을 수행하면 비슷한 사람들끼리 군집을 이룬 결과(label) 을 얻게 됩니다.  

     차원 축소를 하는 일은 feature 를 요약하는 일로 이해를 하면 어떨까요. 사실 빅데이터 분석, 통계를 왜 할까요? 사람이 직관적으로 이해하기에는 너무나 방대해져서 그것을 요약하고, 어떤 의미를 추출하기 위함이라고 생각합니다.  

     차원 축소는 어떤 데이터를 알고리즘에 입력하기 전에 데이터를 변형해서 새로운 training set을 만들어 주는 일입니다. N 차원의 데이터를 N-1 차원으로 축소하는 일, 삼차원 데이터를 평면에 투영시켜 이차원 데이터로 이해하는 일, 문제를 좀 더 단순화 하는 방법이죠. 

   

3. Semi-supervised learning (준지도 학습)

   트레이닝 데이터 셋 이 대부분은 unlabeled 이고 일부만 labeled 되어 있을 때 사용하는 방법으로 지도학습 + 비지도학습 

   - 대표적인 예시가 아이폰의 사람 앨범

     [link](https://support.apple.com/en-us/HT207103)

     

4. Reinforcement Learning (강화학습)

   Agent 라고 불리는 러닝시스템이 있다. 에이전트는 특정 환경에서 목적에 따라 동작을 선택하고 수행하면 그 결과에 따라 보상 혹은 벌을 받는다. 에이전트는 policy 라 부르는 최고의 전략을 스스로 학습해 나간다. 시간이 지날수록 보상을 크게 얻기 위함이다. policy는 에이전트가 어떤 동작을 취해야 하는지 결정하고 언제 주어진 상황에 들어가야 하는가에 대한 선택을 정의한다. 

   - 알파고가 대표적인 예시 



- 데이터가 한번에 주어지는지 점진적으로 주어지는지에 따라

1. Batch learning
2. Online learning



- 보이지 않는 데이터를 일반화 하는 방법에 따라 

1. Instance-based learning

   KNN 에서 새로운 데이터가 주어졌을 때 그 하나의 인스턴스를 기존에 주어지던 인스턴스들과 비교하는 것.

2. Model based learning

   주어진 데이터로 먼저 모델을 만들고, 그 모델이 예측하도록 사용한다. 

   데이터 연구 -> 모델선택 -> Train -> 새로운 케이스를 모델에 넣어서 예측

   

##  Machine Learning Workflow

1. 전처리(Preprocessing)

    실제 세상의 데이터는 Incomplete (불완전, 값이 없음), Noisy(error, outlier), inconsistent(일관성 없는 데이터TF 10) 등의 특징을 지닌다. 

2. 학습(Learning) 

   알고리즘을 이용하여 데이터로 학습을 하고 하이퍼파라미터를 조정하여 모델을 얻습니다. 알고리즘이 문제푸는방법? 이라면 이 방법에 따라 공부한 학생을 모델이라고 할 수 있겠네요. 방법은 같지만 어떤 데이터를 이용했느냐에 따라 다른 모델이 나올 수 있고, 같은 알고리즘에 같은 데이터를이용했지만 hyperparameter setting 에 따라 다른 모델이 나올 수 있습니다. 여기서 algorithm 이 문제 푸는 방법, data 는 학습 방식이나 자료 setting은 세부적인 설정으로 예를 수 있겠습니다. 

   - Algorithm : The way to learn models from data
   - Model :  The product we get ny applying an algorirtrhm to data

   

3. 평가(Evaluationn) 

   - Evaluating & re-learning : an iterative process

     - 알고리즘 선택

     - hyperparameter 조정

     - 모델 학습

     - 모델평가

       이 과정을 반복하여 최적의 모델을 형성합니다.  

       

4. 예측(Prediction) 

   모델에게 unseen data 를 넣어주고 결과를 예측하게 합니다. 



## Challenges of ML

#### Data

​	Insufficient quantity of training data

 - Poor-quality data : training data may have errors, outliers, missing values, and noise
 - irrelevant features : feature engineering 을 이용한 좋은 데이터 feature set을 정하는 일이 중요함. 
    - 많다고 무조건 좋은 것이 아님



#### Algorithm

- Overfitting 

  트레이닝 데이터 셋에 너무 적합한 모델이 나와버려서 일반화가 잘 되지 않는 현상 발생, 트레이닝 데이터에 비해 너무 복잡한 모델이 나온다

  - 해결 방법 
    - trainning data 의 attributes 의 수를 줄이거나 정규화를 하여 parameters를 줄인다. 
    - trainning data 자체를 늘린다. 
    - noise 를 제거한다. (error/outliers)

- Underfitting

  너무 단순한 모델이 나오는 경우, 즉 학습이 잘 안되었다고도 할 수 있죠?

  - 해결방법
    - 더 좋은 모델, 파라미터 수 늘려보기
    - feature engineering 을 통해 더 나은 features로 새로 학습
    - 제약 사항을 조금 더 풀어준다. (e.g., reducing the regularization hyperparameters)





