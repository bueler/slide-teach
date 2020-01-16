all: twopoint.pdf

figures := ivpbvp-crop.pdf resultSHOOT-crop.pdf toyshoot-crop90.pdf \
	   nonconstant-crop90.pdf spyFD-crop.pdf resultFD-crop.pdf  \
	   toyfd-crop90.pdf

twopoint.pdf: twopoint.aux twopoint.tex $(figures)
	pdflatex twopoint

twopoint.aux: twopoint.tex
	pdflatex twopoint

.PHONY: clean

clean:
	@rm -f *.out *.aux *.log *.bbl *.blg *.nav *.snm *.toc *.vrb *~ *.synctex.gz

