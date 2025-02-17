# searchPSD
이 프로그램은 Tkinter 기반 GUI 애플리케이션으로, PNG, JPG, PSD 파일 간 이미지 유사도 비교를 수행합니다.  
멀티프로세싱을 활용하여 효율적으로 이미지를 병렬 처리할 수 있습니다. 주요 기능은 다음과 같습니다:

- PNG 파일 및 폴더 선택 후 분석 가능
- SSIM(구조적 유사도 지수, Structural Similarity Index) 를 활용한 이미지 비교
- PSD 파일 처리 및 레이어 병합 후 비교
- 빠른 비교 및 정밀 비교 모드 지원
- 유사도 결과를 리스트박스에 표시
- 이미지 미리보기 기능 제공
- 멀티프로세싱을 활용한 성능 최적화(멀티프로세싱o 성능최적화x)
- 실시간 진행 상태 업데이트
- 처리 중 작업 중지 가능(bug)
- 이 도구는 대량의 이미지 데이터에서 유사한 이미지를 효과적으로 탐지하는 데 유용합니다.(폴더 탐색 멀티프로세싱으로 수정해야함)

pyinstaller -F -w searchPSD.py --additional-hooks-dir=.

![image](https://github.com/user-attachments/assets/af59cb01-5cbd-4415-a4b6-336ff621a59e)
