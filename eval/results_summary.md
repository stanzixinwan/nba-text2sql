| file | model | method | mode | n_train | exec_acc | exact_acc |
|---|---|---|---|---:|---:|---:|
| full_t5-base_nba_full_test.json | t5-base | full | full_schema | - | 0.0000 | 0.0000 |
| full_t5-base_nba_oracle_test.json | t5-base | full | oracle | 0 | 0.0400 | 0.0000 |
| lora_codet5p-220m_r16_nba_n20_nba_oracle_test.json | codet5p-220m | lora | oracle | 20 | 0.1200 | 0.0600 |
| lora_codet5p-220m_r16_nba_n70_nba_oracle_test.json | codet5p-220m | lora | oracle | 70 | 0.2600 | 0.2400 |
| lora_codet5p-220m_r16_nba_nall_nba_oracle_test.json | codet5p-220m | lora | oracle | all | 0.4600 | 0.4200 |
| lora_codet5p-220m_r16_nba_nall_nba_rag_k1_test.json | codet5p-220m | lora | rag | all | 0.1200 | 0.1200 |
| lora_codet5p-220m_r16_nba_nall_nba_rag_k3_test.json | codet5p-220m | lora | rag | all | 0.0800 | 0.0800 |
| lora_codet5p-220m_r16_nba_nall_nba_rag_k5_test.json | codet5p-220m | lora | rag | all | 0.0800 | 0.0800 |
| lora_codet5p-220m_r16_nba_oracle_test.json | codet5p-220m | lora | oracle | 0 | 0.1000 | 0.0000 |
| lora_t5-base_r16_nba_n20_nba_oracle_test.json | t5-base | lora | oracle | 20 | 0.0400 | 0.0200 |
| lora_t5-base_r16_nba_n70_nba_oracle_test.json | t5-base | lora | oracle | 70 | 0.0800 | 0.0400 |
| lora_t5-base_r16_nba_nall_nba_oracle_test.json | t5-base | lora | oracle | all | 0.1600 | 0.1600 |
| lora_t5-base_r16_nba_nall_nba_rag_k5_test.json | t5-base | lora | rag | all | 0.0200 | 0.0000 |
| lora_t5-base_r16_nba_oracle_test.json | t5-base | lora | oracle | 0 | 0.0400 | 0.0200 |
