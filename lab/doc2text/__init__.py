# coding=utf-8

import os
import mimetypes

import cv2

from .page import Page


acceptable_mime = ["image/bmp", "image/png", "image/tiff", "image/jpeg",
                   "image/jpg", "video/JPEG", "video/jpeg2000"]


FileNotAcceptedException = Exception(
    'The filetype is not acceptable. We accept bmp, png, tiff, jpg, jpeg, jpeg2000, and PDF.'
)


class Document(object):
    def __init__(self, lang=None):
        self.lang = lang
        self.pages = []
        self.processed_pages = []
        self.page_content = []
        self.prepared = False
        self.error = None

    def read(self, path):
        self.filename = os.path.basename(path)
        self.file_basename, self.file_extension = os.path.splitext(self.filename)
        self.path = path
        self.mime_type = mimetypes.guess_type(path)
        self.file_basepath = os.path.dirname(path)

        # If the file is a pdf, split the pdf and prep the pages.
        if self.mime_type[0] in acceptable_mime:
            self.num_pages = 1
            orig_im = cv2.imread(path, 0)
            page = Page(orig_im, 0)
            self.pages.append(page)

        # Otherwise, out of luck.
        else:
            print(self.mime_type[0])
            raise FileNotAcceptedException

    def process(self):
        for page in self.pages:
            new = page
            new.crop()
            new.deskew()
            self.processed_pages.append(new)

    def extract_text(self):
        if len(self.processed_pages) > 0:
            for page in self.processed_pages:
                new = page
                text = new.extract_text()
                self.page_content.append(text)
        else:
            raise Exception('You must run `process()` first.')

    def get_text(self):
        if len(self.page_content) > 0:
            return "\n".join(self.page_content)
        else:
            raise Exception('You must run `extract_text()` first.')
