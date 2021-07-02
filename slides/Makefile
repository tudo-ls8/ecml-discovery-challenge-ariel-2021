MAIN=slides-ariel
PDFLATEX=lualatex -interaction=nonstopmode -halt-on-error

$(MAIN).pdf: $(MAIN).tex $(shell find img -type f)
	$(PDFLATEX) $(@:%.pdf=%)
	$(PDFLATEX) $(@:%.pdf=%)

clean:
	rm -f $(MAIN).pdf \
	  *.aux *.auxlock *.bbl *.blg *.loa *.lof *.log *.nav *.out *.snm *.thm *.toc

.PHONY: clean
