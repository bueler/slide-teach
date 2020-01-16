all: iterative.pdf

iterative.pdf: iterative.aux iterative.tex $(figures)
	pdflatex iterative

iterative.aux: iterative.tex
	pdflatex iterative

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *~ *.synctex.gz

