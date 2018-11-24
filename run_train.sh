python ./run_summarization.py \
	--mode=train \
	--data_path=/project/data/test_for_generator_keyphrase/finished_files/chunked/train_* \
	--vocab_path=/project/data/test_for_generator_keyphrase/finished_files/vocab \
	--log_root=/tmp/test-pointer-generater/log/ \
	--exp_name=myexperiment \
        --coverage=True
