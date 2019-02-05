---
layout: post
title:  "ML Study (4)Unsupervised Learning"
subtitle:   "2019-02-07-study-machinelearning"
categories: study
tags: machinelearning
comments: true
---

4회차(2/7) : 1명 (조문기)(지현이)

- 과제 1 결과공유 

1. Unsupervised Learning : Clustering

   - K-means clustering
   - Hierarchical clustering
   - DBSCAN(Density-based spatial clustering of application with noise)
   - Spectral Clustring

2. Evaluation (Clustering)

   <숙제 :  모델 복습 및 코드 작성>

# REF

* 각자 자료를 조사하면서 출처 혹은 참고하면 좋을만한 자료를 여기에 남겨주세요



# Unsupervised Learning



## Cluster Analysis



## Applicationns of Cluster Analysis 

- Biology

  : 접근 가능한 대량의 유전정보를 분석하기 위해 적용

  예를 들면, 비슷한 기능을 하는 유전자의 그룹을 찾아냄

- Information Retrieval 

  : 검색 결과를 쿼리의 특정 측면을 포착하여 더 작은 수의 그룹으로 쪼갤 때 사용됨

  예를 들면, movie 라는 쿼리를 검색하면 reviews, trailers, theaters 등의 카테고리로 군집화된 웹 페이지를 반환한다. 

- Medicine

  : 질병이나 상태는 빈번하게 다양한 변화를 갖는다. 군집분석은 이런 다양한 하위카테고리를 정의 하는데 쓰임. 

  예를 들어  파킨슨병의 다양한 병을 identify 하는데 쓰임

- Business

  : 비지니스는 현재 혹은 잠재적 고객에 대한 다양한 정보를 수집한다. 

  클러스터링은 추가적 분석이나 마케팅 활동을 위해 고객층을 분류한다. 



## Techniques

- Prototype-based clustering : K-means clustering
- Hierarchical clustering :  Agglomerative heierarchical clustering
- Density-based clustering : DBSCAN
- Spectral clustering



## K-means Clustering

사용자가 설정한 K 개의 그룹으로 나타낸다. 그 그룹들은 centroids가 대표한다. 

#### Algorithm

1. K를 고르고 초기 centroid 를 설정한다. 
2. K 개의 클러스터를 생성 :  각각의 점과 가장 가까운 centroid 를 할당
3. 각각의 클러스터에서 centroid 를 새로 계산한다.
4. 2와 3을 반복한다. centroid 가 바뀌지 않을 때 까지 



#### Objective function 

- SSE(Sum of squred error)

  

#### Outlisers



#### 한계점



## Hierarchical Clustering

- Agglomerative Hierarchical Clustering(bottom-up)

  각 점을 개별 클러스터로 잡고 시작해서 단계마다 가까운 쌍을 합쳐나간다. 

- Divisive Hierarchical Clusterin (top-down)

  전체 점을 포함하는 하나의 클러스터에서 시작하여 단계마다 하나의 원소를 가진 클러스터가 남을 때 까지 쪼개나간다. 

- Dendrogram 덴드로그램 : 계층적 클러스터링은 종종 그래픽으로 트리같이 생긴 다이어그램으로 나타낼 수 있다 

  이 덴드로그램은 클러스터와 하위 클러스터간의 관계와 합쳐진 순서를 보여준다



## Agglomerative Hierarchical Clustering

#### 알고리즘



#### 클러스터간 proximitiy 를 정의하는 방법

- Centroid linkage : 클러스터 센트로이드간의 인접성으로 정의
- Ward's method : 두 클러스터를 병합했을때의 결과로 증가한 SSE 를 가지고 정의 



#### Centrod linkage

- Single linkage : 두 클러스터 간의 가장 가까운 두 점 사이의 근접성
- Complete linkage : 두 클러스터 간의 가장 먼 점 사이의 근접성
- Average linkage : 두 클러스터의 모든 점사이의 근접성의 평균



#### Issues (해결해야 하는 문제점)



## Density-based Clustering



#### Center-based approach



## DBSCAN

- Density-based Spatial Clustering of Applications with Noise



#### 알고리즘





#### DBSCAN 의 parameters

- Eps
- MinPts



#### 방법



#### 파라미터 정하기



#### 장점



#### Issues

- 클러스터들이 넓고 다양한 밀도를 가질 때 문제가 된다. 
- 고차원 데이터에 있어서는 정의하기 힘들다. 
- 모든 데이터의 proximities를 조사해야 하기 때문에 계산 비용이 많이 든다. 



## Spectral Clustering

#### 개념

- Graph Cut
- Cut
- Min cut
- Normalized cut



####  Approach



------

# Cluster Evaluation



## Unsupervised Cluster Evaluation

- Cluster cohesion(commpactness, tightness)

  : 얼마나 서로 가까운지를 측정

- Custer separation(isolation)

  : 얼마나 서로 먼지/ 혹은 얼마나 잘 분리되어 있는지 측정



## Cohension and separation

#### (graph-based view)

#### prototype-based view



## Evaluating Individual Clusters and Objects

#### Silhouette Coefficient(An individual point) (실루엣계수)

#### Silhouette Coefficient(An Cluster) (실루엣계수)



## Determining the Correct Number of Cluster

#### SSE(Sum of Squared Error) 를 이용



## Supervised Cluster Evaluation

#### Classification-oriented

- Entropy
- Precision
- Recall
- F-measure

#### Similarity-oriented







