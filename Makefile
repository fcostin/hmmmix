
all:	demo
.PHONY:	all

demo:	lib/hmmmix/master.py data/transactions.npy
	python -m lib.hmmmix.master $(word 2,$^)
.PHONY:	demo

clean:
	rm -f data/transactions.npy
.PHONY:	clean

data/transactions.npy:	tools/preprocess_transactions.py data/demo_transaction_data_2019_2021.csv
	python $< --count --out $@ $(word 2,$^)
