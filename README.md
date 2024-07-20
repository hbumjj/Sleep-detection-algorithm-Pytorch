# Sleep-Wake-detection-algorithm-using-3-Axis-Accelerometer - Pytorch
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>고령자 수면-각성 감지 알고리즘 개발</title>
</head>
<body>
    <h1>고령자 수면-각성 감지 알고리즘 개발</h1>

    <h2>개요</h2>
    <p>이 프로젝트는 웨어러블 디바이스의 3축 가속도 센서를 활용하여 고령자를 위한 수면-각성 감지 알고리즘을 개발했습니다. 기존 알고리즘들이 고령자의 수면 중 잦은 움직임으로 인해 성능이 저하되는 한계를 극복하고자 했습니다.</p>

    <h2>주요 특징</h2>
    <ul>
        <li>고령자 대상 (평균 연령 72.6세)</li>
        <li>상용 웨어러블 디바이스(ActiGraph wGT3X-BT) 사용</li>
        <li>딥러닝 기법 활용 (Inception Time과 Temporal Convolutional Network)</li>
        <li>고령자의 수면 특성을 반영한 도메인 특화 특징 사용</li>
    </ul>

    <h2>방법론</h2>
    <ol>
        <li>수면다원검사와 웨어러블 디바이스를 이용해 21명의 고령자로부터 데이터 수집</li>
        <li>가속도계 데이터에서 시간 및 주파수 도메인 특징 추출</li>
        <li>Inception Time과 TCN을 결합한 딥러닝 모델 개발</li>
        <li>7겹 교차 검증을 통한 성능 평가</li>
    </ol>

    <h2>결과</h2>
    <ul>
        <li>정확도: 82.66% (±3.63%)</li>
        <li>각성 F1 점수: 0.66 (±0.04)</li>
        <li>코헨의 카파: 0.54 (±0.06)</li>
        <li>수면다원검사로 도출된 수면 파라미터(TST, SE, WASO)와 유의한 상관관계 확인</li>
    </ul>

    <h2>결론</h2>
    <ul>
        <li>제안된 알고리즘이 기존 방법들에 비해 고령자에게 개선된 성능을 보임</li>
        <li>고령자의 실제 수면 질 평가에 적용 가능성 제시</li>
        <li>모든 연령대에 대한 성능 검증을 위한 추가 연구 필요</li>
    </ul>
</body>
</html>