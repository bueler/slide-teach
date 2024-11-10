all: slides.pdf

slides.pdf: slides.aux slides.tex
	pdflatex -shell-escape slides

slides.aux: slides.tex
	pdflatex -shell-escape slides

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *~ *.synctex.gz *.pyg
	@rm -rf _minted-*
