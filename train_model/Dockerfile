FROM python:3.6-alpine
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /opt/train_model/

RUN apk --no-cache --update-cache add gcc gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev libexecinfo-dev
RUN pip install -r /opt/train_model/requirements.txt
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

WORKDIR /opt/train_model/