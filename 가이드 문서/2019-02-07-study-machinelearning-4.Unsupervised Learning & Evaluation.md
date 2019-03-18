---
layout: post
title:  "ML Study (4)Unsupervised Learning"
subtitle:   "2019-02-07-study-machinelearning"
categories: study
tags: machinelearning
comments: true
---

4회차(2/7)** : 1명 (조문기)(지현이)(현오주씨 결석)

- 과제 1 결과공유 

1. Unsupervised Learning : Clustering

   - K-means clustering
   - DBSCAN(Density-based spatial clustering of application with noise)

2. Evaluation (Clustering)

   <숙제 :  모델 복습 및 코드 작성>

# REF

* 각자 자료를 조사하면서 출처 혹은 참고하면 좋을만한 자료를 여기에 남겨주세요



# Unsupervised Learning



## Cluster Analysis

- 군집분석은 대상을 설명하는 데이터에만 기반하여 군집화ㅏㅎㄴ다. 
- 목표는 그룹내 객체들 간의 유사성과 다른 그룹간 차이점을 관찬하는 것이다. 
- 보통 클러스터링은 탐색적 분석 단계에서 사용된다. 



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



- centroid 는 보통 + 로 나타내고 다른 점들은 같은 marker 로 표기한다. 

- Iteration 알고리즘

- K-means 는 colustion 에 수렴한다. 

- 종료 조건 :  

  1, centrid 가 변화하지 않을따

  2.(weak) 점들의 변화율이 1ㅍ퍼센트 미만일때 

- 가장 가까운 센트로이드에 할당하기

  : proximity measure 가 필요하다. 

  - Manhattan(L1)
  - Euclidean(L2)

- 유사도 측정은 상대적으로 간편하다, 각 점과 센트로이드 간의 유사성만 반복적으로 측정하면 되기 때문에 



#### Objective function 

- SSE(Sum of squred error)

  - steps:

    각각 데이터 점의 에러를 계산한다. 그리고 squred error의 토탈썸을 계산한다. 

  - 다양한 클러스터 셋이 주어지면, 가장 작은 SSE를 가진 하나가 최고의 결과를 갖는 형식 

- objective function 을 사용한다. 클러스터의 퀄리티를 확인하기 위해서 

  - Centroid(mean) of the ith cluster

  

- 센트로이드 초기값을 선택한다.(그리 좋지 않을 수도 있음)

- 랜덤으로 초기화 

  - 랜덤으로 선택된 초기 센트로이드값으로 여러번 수행하고 최고의 결과 클러스터 셋을 얻는다. (SSE가 최소값일 때)

- Empty cluster 처리하기.

  만약 어떤 점도 클러스터에 할당되지 않는다면 빈 클러스터가 생성될 수 있다. 

  => 전략은 다른 centroid 를 선택하는 것

  - 현 센트로이드로부터 가장 멀리있는 점을 선택한다. 
  - 이것은 현재 점이 total squred error 에 기여하는 부분을 경감시킨다. 
  - 대체 센트로이드를 SSE가 가장 높은 클러스터에서 선택한다.
  - 이것은 전형적으로 클러스터를 분할하고 전반적인 SSE를 낮춘다.



#### Outlisers

- 클러스터에 영향을 끼친다.
- 아웃라이어가 존재하면, 결과적인 클러스터의 센트로이드는 대표성을 뜨지 않을 수 있고, SSE 또한 높아진다. 
- 그러나 아웃라이어를 제가ㅓ하는 것은 몇몇 적응응에 있어서는 하지 말하야 한다. 
  - 금융분석같은 경우 보통 수이깅 되는 고객들이 아웃라이어인데 없애면 안되는 가장 중요한 파트이기도 하다. 



#### 한계점

- K-means 는 클러스터들이 구가 아닌 모양일 때는 자연 클러스터들을 발견하기에는 어렵다. 

- 클러스터들의 밀도가 다를 떄 

  예를 들어 두개의 작고 밀집된 클러스터와 하나의 널널한 클러스터가 있다다면 

  

## Type of Clusters

#### Hierarchical Clustering

- Agglomerative Hierarchical Clustering(bottom-up)

  각 점을 개별 클러스터로 잡고 시작해서 단계마다 가까운 쌍을 합쳐나간다. 

- Divisive Hierarchical CLusterin (top-down)

  전체 점을 포함하는 하나의 클러스터에서 시작하여 단계마다 하나의 원소를 가진 클러스터가 남을 때 까지 쪼개나간다. 

- Dendrogram 덴드로그램 : 계층적 클러스터링은 종종 그래픽으로 트리같이 생긴 다이어그램으로 나타낼 수 있다 

  이 덴드로그램은 클러스터와 하위 클러스터간의 관계와 합쳐진 순서를 보여준다



## Agglomerative Hierarchical Clustering

#### 알고리즘

> Compute the proximithy matrix, if necessary.
>
> **repeat**
>
> ​	Merge the closest two clusters.
>
> ​	Update the proximity matrix to reflect the proximity between the new cluster and the original clusters.
>
> **until** Only one cluster remains

#### 클러스터간 proximitiy 를 정의하는 방법

- Centroid linkage : 클러스터 센트로이드간의 인접성으로 정의
- Ward's method : 두 클러스터를 병합했을때의 결과로 증가한 SSE 를 가지고 정의 

#### Centrod linkage

- Single linkage : 두 클러스터 간의 가장 가까운 두 점 사이의 근접성
- Complete linkage : 두 클러스터 간의 가장 먼 점 사이의 근접성
- Average linkage : 두 클러스터의 모든 점사이의 근접성의 평균

#### Issues

- 전체적 객관적 기능이 부족하다. 이것은 다양한 기준을 사용하여 각 단계에서 클러스터가 합쳐질 때를 지역적으로 판단한다. 
- 모든 합성의 마지막에 코차원 데이터는 노이즈에 대한 트러블을 유발할 수 있다.
- 계산 및 저장 요구사항의 관점에서 비용이 많이 든다. 



## Density-based Clustering

밀도 기반 클러스터링에서 클러스터는 고밀도 지역으로써 정의된다. 

밀도 기반 클러스터링은 낮은 밀도의 지역으로부터 분리된 고밀도 지역에 위치한다. 



#### Center-based approach

- 밀도를 구하고자 하는 포인트 점을 포함하여 미리 정한 반경 내의 점의 숫자를 세어 데이터셋 내의 특정한 점의 density 를 알아내는 방식
- 점에 분류는 중심기반 밀도에 따른다
  - Core points : 밀도 기반 클러스터 내에서 만약 특정 점 중심으로 특정 거리함수나 파라미터 혹은 Eps 에 따라 정의된 이웃 점들의 수가 특정 한계(MinPts)를 넘었을 떄 그 특정 점을 core points 라 칭한다. 
  - Border points : core points 의 이웃 내에 존재하는 점들
  - Noise points : core 도 border 도 아닌 점



## DBSCAN

- Density-based Spatial Clustering of Applications with Noise
  - 두개의 core points 가 충분히 가깝다면,(Eps 내,) 같은 군집
  - border points 가 core points와 충분히 가깝다면, 같은 군집
  - noise points 는 버린다. 



#### 알고리즘

> Label all points as core,border, or noise points
>
> Eliminate noise points.
>
> Pull an edge between all core points that are within Eps of each other.
>
> Make each group of connected core points into a separate cluster
>
> Assign each border point to one of the clusters of its associate core points

#### DBSCAN 의 parameters

- Eps
- MinPts

이 접근법은 하나의 점으로부터 k번째 가까운 이웃까지 거리의 행동을 살펴보는 것이다. (k-dist)

- 한 점이 클러스터에 속하면 k-dist 는 작고, 그렇지않으면 크다

  (Noise points 같은 경우 k-dist 가 큼)

- 파라미터 정하기
  - 모든 데이터 포인트 대상으로 k에 대해 k-dist 를 계산한다.
  - 오름차순으로 정렬하고, plot을 그린다. 
  - 갑자기 k-dist가 급격히 증가하는 순간이 적당한 Eps 이다. 
- 만약 k 가 너무 작으면 noise 나 아웃라이어들이 클러스터로 속하게 되고
- 너무 크면 작은 클러스터들이 (k보다 사이즈가 작은 것들) noise 된다.
- 원래의 DBSCAN 알고리즘은 2차원 데이터에 k=4를 보통 썼다. 



장점은 미리 k를 구할 필요가 없고, 모양이 원이 아니라도 가능하다는 것



#### Issues

- 클러스터들이 넓고 다양한 밀도를 가질 때 문제가 된다. 
- 고차원 데이터에 있어서는 정의하기 힘들다. 
- 모든 데이터의 proximities를 조사해야 하기 때문에 계산 비용이 많이 든다. 



## Spectral Clustering



------

# Cluster Evaluation



## Unsupervised Cluster Evaluation

- Cluster cohesion(commpactness, tightness)

  : 얼마나 서로 가까운지를 측정

- cluster separation(isolation)

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







