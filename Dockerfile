FROM tiangolo/uvicorn-gunicorn:python3.10-slim

LABEL maintainer="https://github.com/Zetroc99"

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./api/api.py /code/app

CMD ["uvicorn", "api.api:api", "--host", "0.0.0.0", "--port", "5001"]
