SHELL=/bin/bash
CONDA_ACTIVATE=source $(shell conda info --base)/etc/profile.d/conda.sh && conda activate ariel

all: $(patsubst %.tex,%.pdf,$(wildcard *.tex))

z_paa.pdf: z_paa.tex .Z_PAA
	pdflatex $<
.Z_PAA: z_paa.py
	$(CONDA_ACTIVATE) && python $< && touch $@

architecture.pdf: architecture.tex
	pdflatex $<

.PHONY: all
