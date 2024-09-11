all: polybasics.pdf

figures :=

polynonewt.pdf: polynonewt.tex $(figures)
	pdflatex polynonewt
	pdflatex polynonewt

polybasics.pdf: polybasics.tex $(figures)
	pdflatex polybasics
	pdflatex polybasics

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *.synctex.gz *~

