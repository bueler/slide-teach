all: polybasics.pdf

figures :=

polybasics.pdf: polybasics.tex $(figures)
	pdflatex polybasics
	pdflatex polybasics

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *.synctex.gz *~

