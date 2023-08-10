# --- base image
FROM python:3.10 as base

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["python"]

COPY . .

CMD ["/app/src/train.py"]
