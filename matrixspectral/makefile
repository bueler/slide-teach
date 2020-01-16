all: slides.pdf

slides.pdf: slides.aux slides.tex
	pdflatex slides

slides.aux: slides.tex
	pdflatex slides

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *~ *.synctex.gz

