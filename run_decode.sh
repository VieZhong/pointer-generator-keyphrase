python ./run_summarization.py \
	--mode=decode \
	--data_path=/project/data/test_for_generator_keyphrase/finished_files/chunked/val_* \
	--vocab_path=/project/data/test_for_generator_keyphrase/finished_files/vocab \
	--log_root=/tmp/test-pointer-generater/log/ \
	--exp_name=myexperiment
