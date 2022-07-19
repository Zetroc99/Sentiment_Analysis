curl -X 'POST' \
  'http://localhost:5001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "The flight was terrible"}'