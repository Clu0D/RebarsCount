token for jupyter:
`stardist`

FastAPI server:
`docker compose up --build api`

Jupyter:
`docker compose up --build jupyter`

API endpoints:
`GET /health`
`POST /predict_points` with multipart field `file`
`POST /predict_zones` with multipart field `file`

Example:
`curl -X POST http://localhost:8001/predict_points -F "file=@test.png"`
