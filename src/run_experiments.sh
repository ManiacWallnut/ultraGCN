#!/bin/bash

# 데이터셋 목록을 배열로 정의
configs=("ultragcn_amazonbooks_m1.ini" "ultragcn_gowalla_m1.ini" "ultragcn_movielens1m_m1.ini" "ultragcn_yelp18_m1.ini")

# 로그 파일 이름
log_dir="../logs"
mkdir -p $log_dir  # 로그 디렉토리 생성

# 각 데이터셋에 대해 실험을 반복
for config in "${configs[@]}"
do
    # 데이터셋에 맞는 config 파일 경로 설정
    config_file="../config/${config}"
    
    # config 파일이 존재하는지 확인
    if [ ! -f "$config_file" ]; then
        echo "Config file for $config not found!"
        continue
    fi

    # 로그 파일 경로 설정
    log_file="${log_dir}/${config}_experiment.log"

    # 실험 시작 메시지
    echo "Starting experiment for $config using config file $config_file..." | tee -a $log_file

    # 모델 학습 실행하고 표준 출력과 로그 파일에 결과 저장
    python main.py --config_file $config_file | tee -a $log_file

    # 실험 완료 메시지
    echo "Experiment for $config completed." | tee -a $log_file
    echo "---------------------------------------" | tee -a $log_file
done
