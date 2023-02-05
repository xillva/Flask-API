FROM continuumio/anaconda3
COPY ./Flask_Deployment /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirement.txt
CMD ["python","flask_api.py"]