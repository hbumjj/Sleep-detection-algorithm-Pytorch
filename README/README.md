# 고령자 수면-각성 감지 알고리즘 개발

## 개요

이 프로젝트는 웨어러블 디바이스의 3축 가속도 센서를 활용하여 고령자를 위한 수면-각성 감지 알고리즘을 개발했습니다. 기존 알고리즘들이 고령자의 수면 중 잦은 움직임으로 인해 성능이 저하되는 한계를 극복하고자 했습니다.

## 주요 특징

- 고령자 대상 (평균 연령 72.6세)
- 상용 웨어러블 디바이스(ActiGraph wGT3X-BT) 사용
- 딥러닝 기법 활용 (Inception Time과 Temporal Convolutional Network)
- 고령자의 수면 특성을 반영한 도메인 특화 특징 사용

## 방법론

1. 수면다원검사와 웨어러블 디바이스를 이용해 21명의 고령자로부터 데이터 수집
2. 가속도계 데이터에서 시간 및 주파수 도메인 특징 추출
3. Inception Time과 TCN을 결합한 딥러닝 모델 개발
4. 7 fold validation을 통한 성능 평가

## 딥러닝 모델 구조
<div align="center">
  <img src="model.tif" alt="model structure" width="50%">
</div>

<p align="center">
그림 1: Inception time과 Temporal Convolutional Network를 기반으로 한 딥러닝 모델 구조
</p>

## 결과

- Accuracy: 82.66% (±3.63%)
- Wake F1 score: 0.66 (±0.04)
- Cohen's Kappa: 0.54 (±0.06)
- 수면다원검사로 도출된 수면 파라미터(TST, SE, WASO)와 유의한 상관관계 확인

## Attention 기반 시각화
<div align="center">
  <img src="attention.tif" alt="Attention" width="50%">
</div>


<p align="center">
그림 2: Attention heat map을 이용한 모델 결과 시각화
</p>

## 결론

- 제안된 알고리즘이 기존 방법들에 비해 고령자에게 개선된 성능을 보임
- 고령자의 실제 수면 질 평가에 적용 가능성 제시
- 모든 연령대에 대한 성능 검증을 위한 추가 연구 필요