LIBTRELLIS=lib/hmmmix/trellis/libtrellis.cpython-38-x86_64-linux-gnu.so

all:	demo
.PHONY:	all

demo:	lib/hmmmix/main.py data/transactions.npy $(LIBTRELLIS)
	python -m lib.hmmmix.main $(word 2,$^)
.PHONY:	demo

demo-profile:	lib/hmmmix/main.py data/transactions.npy $(LIBTRELLIS)
	python -m lib.hmmmix.main --profile $(word 2,$^)
.PHONY:	demo-profile

test:
	python -m pytest -v .
.PHONY: test



clean:
	rm -f data/transactions.npy
.PHONY:	clean

$(LIBTRELLIS): setup.py lib/hmmmix/trellis/libtrellis.pyx
	python setup.py build_ext --inplace

data/transactions.npy:	tools/preprocess_transactions.py data/demo_transaction_data_2019_2021.csv
	# python $< --count --descr=jah --out $@ $(word 2,$^)
	python $< --count --out $@ $(word 2,$^)
