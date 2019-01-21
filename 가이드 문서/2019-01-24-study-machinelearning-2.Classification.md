---
layout: post
title:  "ML Study (2)Classification"
subtitle:   "2019-01-24-study-machinelearning"
categories: study
tags: machinelearning
comments: true
---

* 2회차 (1/24) : 

   KNN(전은정,조문기)

  N-B(신광욱,유응규)

  SVM(지현이,현오주)

  Regression(나지혜,이지인)

  Decistion Tree(이가경)	

* 각자 맡으신 부분의 수업을 준비해 주세요!

* 어떤 내용을 설명해주시면 좋을지 아래 목록을 적어 두었습니다.

* (예시일 뿐이니 각자 하실 수 있는 범위에서 설명해주시면 됩니다.)

* 자료 조사했을 때 유용한 부분은 REF에 달아주시고 각자 맡은 항목 내용을 읽어서 알아볼 수 있을 정도로만 정리해서 공유해주시면 됩니다. 

* 준비하다가 어려운 부분이 있으시면 연락주세요!

# REF

* 각자 자료를 조사하면서 출처 혹은 참고하면 좋을만한 자료를 여기에 남겨주세요



* [여기에는 제목을  ]  사이간격을 없애고 (여기엔 링크를 )
* [관련 youtube 설명영상](유투브 주소)

------

# Supervised Learning

#### Classicication : Predictt the **categorical** class labels

#### Regression  :  Predict a target **numeric** value



## KNN(k-Nearest Neighbors)

- k와 Nearest Neighbors의 의미

- 작동 알고리즘

- Distance Metrics

  - Euclidian Distance
  - Manhattan Distance
  - Chebyshev Distance
  - Minkowski Distance

  

## Naive Bayes

- 통계학 기초( Probability Basics ) - 만약에 통계 기초가 필요할 경우에

  - Prior probability
  - Conditional probability
  - Joint probbability
  - Independence

- Bayes' Theorem (law or rule)

- Probabilistic Model for Classification

  - chain rule 개념정도,,, 

- Naive Bayes

  

## Support Vector Machines

- Support Vectors 개념 정의, 용어 설명

- Maximum-margin Hyperplane

- Hard Margin

- Soft Margin

- Kernel Trick

  - Kernel Functions

- Multiclass SVM

  

## Linear Regression

- simple/ multiple linear regression 

- MSE(Mean Squared Error)/cost function 설명

- 세타의 값을 찾는 방법 - gradient descent/ 

- learning rate

- challenges

- Convex Function 

- Polynominal regression 

- Regularized Lineaer model

  - Norm 
  - Ridge Regression
  - Lasso Regression 

  

## Logistic Regression 

- 정의, 사용목적, 
- 방정식, logistic function,  prediction, training cost function 
- Multinomial Logistic Regression (softmax regression )

## Decision Trees

* Decision Tree 는 분류, 회귀 모두에 쓰이는 지도학습 모형이다. 

* DT 의 목표는 특정 모델을 생성하는 것인데, 이 모델은 데이터의 feaeture로 부터 추론된 간단한 결정규칙을 학습함에 따라서 목표변수의 값을 예측한다. 

* Which one?

  - If the attributes are adequate, it is always possible to construct a decision tree that correctly classifies each object in the training set.

    만약 속성이 정확하다면 학습데이터 셋에 있는 각각의 객체를 정확하게 구분해내는 의사결정트리를 만드는 것이 가능하다.  

  - Usually there are many such correct decision tress. 

    보통 그런 맞는 트리들이 존재한다. 

  - The essence of induction is to move beyond the training set, to constuct a decision tree that correctly classifies not only objects from the training set but unseen objects as well. 

    트레이넹 세트 뿐만아니라 새로운 데이터 또한 잘 구분해 내기 위한 DT를 만들기 위해서, 연역의 핵심은 트레이닝 셋트를 넘어서는 것이다. 

  - Given a choice between two decision trees, each of which is correct over the training set, select the simpler one that is more likely to capture structure inherent in the problem

    만약 두 개의 주어진 트리가 있을때, 두둘다는 모두 학습 데이이터 셋에 대해 올바른 결과를 니다. 이때 더 단순한 모델을 고른다. 그 모델은 문제에 내제된 구조를 포착할 가능성이 더욱 높다. 

* ID3

  - 정확하게 구분해낼 수 있는 모든 가능한 DT를 생성하고 가장 단훈한 모델을 찾아ㅏ내는 것은 오직 작은 일에만 가능하다. 
  - ID3는 위에서 아래로 만들면서 DT를 학습한다. 
    - "Which attribute is the best classifier?"
  - 정보는 얻는다. 

* Entropy

* Information Gain 



