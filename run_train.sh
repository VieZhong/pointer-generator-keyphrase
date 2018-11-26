python ./run_summarization.py \
	--mode=train \
	--data_path=/project/data/test_for_generator_keyphrase/finished_files/chunked/train_* \
	--vocab_path=/project/data/test_for_generator_keyphrase/finished_files/vocab \
	--stop_words_path=/project/data/stopword/stopword_en.txt \
        --log_root=/tmp/test-pointer-generater/log/ \
	--exp_name=cooccurenceexperiment \
        --coverage=True \
        --max_keyphrase_num=10
