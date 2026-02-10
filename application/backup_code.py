# train
#     
# clip 없이 테스트
    # clip_bounds_dict = {}
    # for col in g_vars.FEATURES:
    #     # 1. 일단 콴타일로 범위를 계산합니다.
    #     lower = float(setting_user_df_chunk[col].quantile(0.1))
    #     upper = float(setting_user_df_chunk[col].quantile(0.9))
        
    #     # 2. [핵심 수정] 속도와 관련된 지표들은 '움직임 없음(0)'을 허용해야 합니다.
    #     # 학습 데이터에 움직임만 있더라도, 추론 시 정지 상태를 위해 min을 0으로 강제합니다.
    #     if col in ['speed', 'speed_var', 'jerk_std', 'straightness']:
    #         # straightness는 보통 1이 최소지만 안전하게 0이나 데이터 최소값 중 작은 쪽 선택
    #         lower = 0.0 
        
    #     # 3. 각 지표별 특성에 따른 하한선 보정 (필요시)
    #     # turn, acc 등은 음수가 가능하므로 콴타일 그대로 사용
        
    #     clip_bounds_dict[col] = {"min": lower, "max": upper}
    #     setting_user_df_chunk[col] = setting_user_df_chunk[col].clip(lower, upper)

        
    
    # # 전역 변수 업데이트
    # g_vars.CLIP_BOUNDS = clip_bounds_dict
    # update_parameters({"CLIP_BOUNDS" : clip_bounds_dict})
    
    # print(f"{json.dumps(clip_bounds_dict, indent=2)} Save")




    # # ===== seq 생성 ======
    # user_train_seq, _ = points_to_features(df_chunk=user_train_df, seq_len=g_vars.SEQ_LEN, stride=g_vars.STRIDE, log_queue=self.log_queue)

    # user_val_seq, _ = points_to_features(df_chunk=user_val_df, seq_len=g_vars.SEQ_LEN, stride=g_vars.STRIDE, log_queue=self.log_queue)


# inference

        # if g_vars.CLIP_BOUNDS:
        #     for col, b in g_vars.CLIP_BOUNDS.items():
        #         if col in df_features.columns:
        #             df_features[col] = df_features[col].clip(lower=b['min'], upper=b['max'])



########

        # clip_bounds_dict = {}
        # for col in g_vars.FEATURES:
        #     # 스케일링 된 값에서의 10%, 90% 지점 찾기
        #     lower = float(chunks_scaled_df[col].quantile(0.01))
        #     upper = float(chunks_scaled_df[col].quantile(0.99))

        #     clip_bounds_dict[col] = {"min": lower, "max": upper}
        #     chunks_scaled_df[col] = chunks_scaled_df[col].clip(lower, upper)

        # g_vars.CLIP_BOUNDS = clip_bounds_dict
        # update_parameters({"CLIP_BOUNDS": clip_bounds_dict})

        # 4. 시퀀스 생성을 위해 최종적으로 다시 numpy로 변환 (필요시)