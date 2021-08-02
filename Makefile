LIBTRELLIS=lib/hmmmix/trellis/libtrellis.cpython-38-x86_64-linux-gnu.so

DEMO_ARGS=--obj-cutoff -3800.0

all:	demo
.PHONY:	all

demo:	lib/hmmmix/main.py data/transactions.npy $(LIBTRELLIS)
	python -m lib.hmmmix.main $(DEMO_ARGS) $(word 2,$^)
.PHONY:	demo

demo-profile:	lib/hmmmix/main.py data/transactions.npy $(LIBTRELLIS)
	python -m lib.hmmmix.main --profile $(DEMO_ARGS) $(word 2,$^)
.PHONY:	demo-profile

demo-periodic-bernoulli: data/transactions.npy
	python -m lib.hmmmix.model.periodic_bernoulli.demo $<
.PHONY: demo-periodic-bernoulli

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
