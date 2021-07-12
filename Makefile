
all:	data/transactions.npy
.PHONY:	all

clean:
	rm -f data/transactions.npy
.PHONY:	clean

data/transactions.npy:	tools/preprocess_transactions.py data/labelled_transactions_inscope_2019_2021.csv
	python $< --out $@ $(word 2,$^)
