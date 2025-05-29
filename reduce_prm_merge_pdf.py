# FIXME: merge with LAr_PrM.py

import os
from glob import glob
from PyPDF2 import PdfMerger

import LAr_PrM as pm

pdf_dir = pm.diagnostic_output_dir(verbose=True)

def pdf_merge():
    ''' Merges all the pdf files in a given directory '''
    merger = PdfMerger()
    pdf_list = glob(os.path.join(pdf_dir, "2025*.pdf"))
    pdf_list = sorted(pdf_list)
    #[merger.append(pdf) for pdf in allpdfs]
    [merger.append(pdf) for pdf in pdf_list]

    out_pdf = os.path.join(pdf_dir, "diagnostic_all.pdf")
    with open(out_pdf, "wb") as new_file:
        merger.write(new_file)

if __name__ == "__main__":
    pdf_merge()
