import random
import matplotlib.pyplot as plt
'''
사분원 그리기 
    목표 : 몬테 카를로 시뮬레이션을 사용한 반지름이 1인 사분원을 그려보자. 
    조건 : x^2 + y^2 <= 1
          임의의 X(0 <= X <= 1), Y(0 <= Y <= 1) 점이 위 조건을 만족하면다면 
          사분원 범위에 들어간다
    랜덤 함수로 위 조건을 만족하는 무수한 X, Y를 가져와서 사분원에 찍는다면 사분원은 언젠가는 꽉차게 될것이다
    
    그럼 이를 수식으로 표현해보면
        - 임의의 점이 사분원 안에 들어가 확률 P는?
            P(X, Y) = pi / 4 / 1 = pi / 4
        - N번 시도해보는 회수가 Trials, 그중 조건에 만족하는 회수를 Hits 이라고 하자
        그럼 이렇게 표현가능하다
            N Hits / N Trials ~= pi / 4
            pi ~= 4 * N Hits / N Trials
            즉, pi는 N번 시도중에 참인 조건의 비례에 4를 곱한것의 근사치임을 알수 있다
    - 당연히 많이 시도를 할수록 결과는 목표 값에 가까워 진다
    - 랜덤 함수의 성능에 의존적이다
    - matplotlib으로 사분원을 그려보면 길이 1을 근소하게 벗어가는 점도 찍히는것으로 보이는데
      이건 matplot의 오차일까? 단순 계산을 해봐도 벗어나는데..?
    
    참조 
        http://phya.snu.ac.kr/~kclee/lects/lect06/lect06.htm
        http://www.playsw.or.kr/repo/cast/109
'''
def draw_quarter_circle(Trials):
    Hits=0
    X = []
    Y = []
    for i in range(Trials):
        x = random.uniform(0, 1.0)
        y = random.uniform(0, 1.0)
        if (x**2 + y**2) <= 1:
            Hits = Hits + 1
            X.append(x)
            Y.append(y)
    plt.plot(X, Y, 'ro')
    plt.axis([0, 1.5, 0, 1.5])
    plt.show()
    print("value of pi? ",4 * Hits / Trials)

draw_quarter_circle(1000000)