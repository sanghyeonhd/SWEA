````markdown
## beamWeldFoam

### 개요
고에너지 밀도 첨단 제조 공정을 연구하기 위해 제시된 확장 가능한 오픈 소스 VOF(Volume-of-Fluid) 솔버 beamWeldFoam입니다. 이 구현에서 금속 기판과 보호 가스상은 비압축성으로 처리됩니다. 솔버는 금속 기판의 융합/용융 상태 전환을 완벽하게 포착합니다. 기판의 증발의 경우, 증발 상태 전환으로 인한 명시적인 부피 팽창은 무시되는 대신, 현상론적 반동 압력 항이 증발 현상으로 인한 운동량 및 에너지 장에 대한 기여를 포착하는 데 사용됩니다. beamWeldFoam은 또한 표면 장력 효과, 표면 장력의 온도 의존성 (마랑고니) 효과, 용융/융합 (및 증발)로 인한 잠열 효과, 부시네스크 근사를 사용하여 상의 열팽창으로 인한 부력 효과, 응고로 인한 운동량 감쇠, 그리고 입사하는 레이저/전자빔 열원의 대표적인 열원 설명을 포착합니다. 이 열원은 아크 용접 공정을 나타내도록 수정할 수도 있습니다.

솔버 접근 방식은 [OpenCFD Ltd.](http://openfoam.com/)에서 개발한 단열 이상 유동 interFoam 코드를 기반으로 합니다. beamWeldFoam의 목표 적용 분야는 다음과 같습니다.

* 레이저 용접
* 전자빔 용접
* 아크 용접
* 적층 제조

### 설치

현재 코드 버전은 [OpenFoam6 라이브러리](https://openfoam.org/version/6/)를 사용합니다. 이 코드는 Ubuntu 환경에서 개발 및 테스트되었지만 OpenFoam을 설치할 수 있는 모든 운영 체제에서 작동해야 합니다. beamWeldFoam 솔버를 설치하려면 먼저 이 페이지 ([OpenFoam 6 설치](https://openfoam.org/download/6-ubuntu/))의 지침에 따라 OpenFoam 6 라이브러리를 설치하십시오.

OpenFoam10과 함께 사용하려면 OF10 브랜치를 선택하십시오.

그런 다음 쉘 터미널에서 작업 폴더로 이동하여 git 코드 저장소를 복제하고 빌드합니다.

```bash
$ git clone [https://github.com/tomflint22/beamWeldFoam.git](https://github.com/tomflint22/beamWeldFoam.git) beamWeldFoam
$ cd beamWeldFoam/applications/solvers/beamWeldFoam/
$ wclean
$ wmake
````

아래 설명된 튜토리얼 케이스를 사용하여 설치를 테스트할 수 있습니다.

### 튜토리얼 케이스

직렬 모드에서 튜토리얼을 실행하려면 다음을 수행하십시오.

```bash
오래된 시뮬레이션 파일을 삭제하십시오 (예:):
$ rm -r 0* 1* 2* 3* 4* 5* 6* 7* 8* 9*
그 다음:
$ cp -r initial 0
$ blockMesh
$ setFields
$ beamWeldFoam
```

`setFields` 명령 이후 MPI를 사용하여 병렬 배포를 하려면 다음을 수행하십시오.

```bash
$ decomposePar
$ mpirun -np 6 beamWeldFoam -parallel >log &
```

6개의 코어에 배포하는 경우입니다.

#### 갈륨 용융 케이스

용융 및 응고가 관련된 열 및 물질 전달의 일반적인 검증 사례는 밀폐된 용기에서 갈륨 용융을 시뮬레이션하는 것입니다. 이 예에서는 beamWeldFoam 솔버를 사용하여 갈륨의 용융과 그에 따른 부력으로 인한 흐름을 시뮬레이션합니다. 시간이 지남에 따라 계산 영역의 왼쪽 뜨거운 벽은 국소적으로 갈륨을 용융시킵니다. 용융 부피가 증가함에 따라 뜨거운 액체 갈륨이 상승하여 액체 내에 와류 구조를 생성하므로 부력 구동 흐름이 지배적입니다. 예측된 용융 프로파일은 수치적 및 실험적으로 다른 곳에서 보고된 결과와 매우 잘 일치합니다\[1].

#### 마랑고니 흐름 (Sen and Davies) 케이스

솔버의 또 다른 유용한 검증 사례는 2D 공동이 부분적으로 채워져 상 사이의 계면이 처음에는 평평한 경우입니다. 그런 다음 도메인 전체에 온도 기울기가 발생합니다. 이 온도 기울기는 표면 장력의 온도 의존성(일명 마랑고니 흐름)으로 인해 계면에 접하는 흐름을 유도합니다. 이 경우 자유 표면 변형에 대한 분석적 정상 상태 해가 존재합니다\[2]. beamWeldFoam 솔버와 분석적 해 사이에는 매우 우수한 일치가 관찰됩니다.

#### 아크 용접 케이스

이 예에서는 아크 용접 공정을 대표하는 알루미늄 기판에 표면 열 플럭스가 적용됩니다. 이 시나리오에서는 아르곤 가스의 두 영역 사이에 금속 기판이 도메인에 존재합니다. 열원은 t=0s에 적용되고, t=0.25s에 전력이 감소하기 시작하여 t=0.35s에 열원이 완전히 꺼집니다. 열원이 꺼진 직후 도메인은 완전히 응고됩니다. 마랑고니 구동 흐름의 효과는 이 예에서 명확하게 볼 수 있습니다. 표면 흐름은 온도가 높은 영역에서 온도가 낮은 영역으로 구동됩니다 (온도에 따른 표면 장력 감소로 인해). 또한 용접 풀이 도메인을 완전히 관통한 후에는 표면 장력이 물질이 기판 아래쪽에서 떨어지는 것을 방지합니다.

#### 빔 용접 케이스

이 예에서는 beamWeldFoam을 적용하여 티타늄 합금 기판의 파워 빔 용접을 시뮬레이션합니다. 이 경우 레이저 빔으로 용접된 Ti6Al4V 맞대기 이음매를 시뮬레이션하고 결과를 실험 연구\[3]와 비교하여 검증합니다.

### 알고리즘

솔버는 처음에 메쉬를 로드하고, 필드 및 경계 조건을 읽어들이고, 특정 메쉬 정보를 배열에 읽어들이고 (열원 적용을 위해), 난류 모델을 선택합니다 (지정된 경우). 그런 다음 주요 솔버 루프가 시작됩니다. 첫째, 수치적 안정성을 보장하기 위해 시간 간격이 동적으로 수정됩니다. 다음으로, 이상 유체 혼합물 속성 및 난류량이 업데이트됩니다. 이산화된 상 분율 방정식은 다차원 명시적 해법(MULES)을 사용하는 사용자 정의 하위 시간 단계 수(일반적으로 3)에 대해 풀립니다([MULES](https://openfoam.org/release/2-3-0/multiphase/)). 이 솔버는 OpenFOAM 라이브러리에 포함되어 있으며 정의된 경계(α1의 경우 0과 1)를 사용하여 쌍곡선 대류 수송 방정식의 보존적 해를 수행합니다. 업데이트된 상 필드를 얻으면 프로그램은 압력-속도 루프에 들어가고, 이 루프에서 p와 u가 교대로 수정됩니다. 이 루프에서 T도 풀려서 부력 예측이 U 및 p 필드에 대해 정확하도록 합니다. 압력 및 속도 필드를 순차적으로 수정하는 프로세스를 연산자 분할을 통한 압력 암시적 방법(PISO)이라고 합니다. OpenFOAM 환경에서 PISO는 각 시간 단계에서 여러 번 반복됩니다. 이 프로세스를 병합된 PISO - 압력-연결 방정식의 반-암시적 방법(SIMPLE) 또는 압력-속도 루프(PIMPLE) 프로세스라고 하며, 여기서 SIMPLE은 반복적인 압력-속도 해법 알고리즘입니다. PIMPLE은 사용자가 지정한 반복 횟수 동안 계속됩니다. 주요 솔버 루프는 프로그램이 종료될 때까지 반복됩니다. 시뮬레이션 알고리즘의 요약은 아래에 제시되어 있습니다.

  * beamWeldFoam 시뮬레이션 알고리즘 요약:
      * 시뮬레이션 데이터 및 메쉬 초기화
      * t \< t\_end인 동안 다음을 반복합니다.
        1.  안정성을 위해 delta\_t 업데이트
        2.  상 방정식 하위 사이클
        3.  열원 적용을 위한 인터페이스 위치 업데이트
        4.  유체 속성 업데이트
        5.  PISO 루프
            1.  u 방정식 구성
            2.  에너지 전달 루프
                1.  T 방정식 풀기
                2.  유체 분율 필드 업데이트
                3.  잠열로 인한 소스 항 재평가
            3.  PISO
                1.  면 플럭스 얻고 수정
                2.  p-Poisson 방정식 풀기
                3.  u 수정
        6.  필드 쓰기

갈륨 용융 및 Sen and Davies 케이스의 두 가지 샘플 튜토리얼 케이스는 문헌에서 이용 가능한 실험 및 분석 데이터와 매우 잘 일치하며 beamWeldFoam 구현의 검증 사례 역할을 합니다.

### 라이선스

OpenFoam 및 확장적으로 beamWeldFoam 애플리케이션은 [GNU 일반 공중 사용 허가서 버전 3](https://www.gnu.org/licenses/gpl-3.0.en.html)에 따라서만 무료 및 오픈 소스로 라이선스가 부여됩니다. OpenFOAM의 인기 비결 중 하나는 사용자가 GPL 조건 내에서 소프트웨어를 자유롭게 수정하고 재배포할 수 있으며 지속적인 무료 사용 권한을 부여받는다는 것입니다.

### 감사의 말씀

이 연구는 유럽 지역 개발 기금 및 I-Form 산업 파트너의 공동 지원을 받는 Science Foundation Ireland (SFI) 보조금 16/RC/3872와 ''Cobalt-free Hard-facing for Reactor Systems'' 보조금 EP/T016728/1에 따라 영국 공학 및 물리 과학 연구회 (EPSRC)의 아낌없는 지원을 받았습니다.

### 본 연구 인용 방법

본 연구에서 beamWeldFoam을 사용하신 경우, 다음을 사용하여 본 연구를 인용해 주십시오.

Thomas F. Flint, Gowthaman Parivendhan, Alojz Ivankovic, Michael C. Smith, Philip Cardiff,
beamWeldFoam: 고에너지 밀도 융합 및 증발 유도 공정의 수치 시뮬레이션,
SoftwareX,
18권,
2022,
101065,
ISSN 2352-7110,
https://doi.org/10.1016/j.softx.2022.101065

### 참고 문헌

1.  Kay Wittig and Petr A Nikrityuk 2012 IOP Conf. Ser.: Mater. Sci. Eng. 27 012054
2.  Sen, A., & Davis, S. (1982). Steady thermocapillary flows in two-dimensional slots. Journal of Fluid Mechanics, 121, 163-186. doi:10.1017/S0022112082001840
3.  Sabina L. Campanelli, Giuseppe Casalino, Michelangelo Mortello, Andrea Angelastro, Antonio Domenico Ludovico, Microstructural Characteristics and Mechanical Properties of Ti6Al4V Alloy Fiber Laser Welds

```
```

