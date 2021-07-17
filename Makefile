LIBLATTICE=lib/hmmmix/lattice/liblattice.cpython-38-x86_64-linux-gnu.so

all:	demo
.PHONY:	all

demo:	lib/hmmmix/main.py data/transactions.npy $(LIBLATTICE)
	python -m lib.hmmmix.main $(word 2,$^)
.PHONY:	demo

clean:
	rm -f data/transactions.npy
.PHONY:	clean

$(LIBLATTICE): setup.py lib/hmmmix/lattice/liblattice.pyx
	python setup.py build_ext --inplace

data/transactions.npy:	tools/preprocess_transactions.py data/demo_transaction_data_2019_2021.csv
	python $< --count --descr=jah --out $@ $(word 2,$^)
