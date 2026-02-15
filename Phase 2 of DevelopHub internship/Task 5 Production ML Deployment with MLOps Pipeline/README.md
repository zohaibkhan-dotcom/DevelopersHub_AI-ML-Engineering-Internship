# Production ML Deployment with MLOps Pipeline

## Project Overview

This project demonstrates a complete, industry-level, and production-ready Machine Learning deployment with an MLOps pipeline. The objective is to take a trained Machine Learning model (e.g., a customer churn prediction model) and deploy it in a scalable system. It includes a REST API for inference, model versioning, logging, monitoring, Docker containerization, and a CI/CD ready structure, simulating a real-world ML production environment.

## Architecture Diagram (Text Explanation)

The system architecture is designed around a FastAPI application serving as the core ML inference service. This application loads a pre-trained model and exposes a `/predict` endpoint for real-time predictions. It also includes `/health` for service status and dedicated endpoints for uploading various file types (CSV, TXT, Image, IPYNB). All incoming requests, predictions, and errors are logged. The application is containerized using Docker for consistent deployment across different environments. Model versions are managed through a structured directory, allowing for easy switching and updates.

```
+-----------------------+
|      Client/User      |
+-----------+-----------+
            |
            | HTTP/HTTPS Requests
            |
+-----------v-----------+
|      FastAPI App      |
| (uvicorn, Pydantic)   |
+-----------+-----------+
| - /health             |
| - /predict            |
| - /upload/csv         |
| - /upload/txt         |
| - /upload/image       |
| - /upload/ipynb       |
+-----------+-----------+
            |
            | Logs (File)
            |
+-----------v-----------+
|      Logging System   |
| (Python logging)      |
+-----------------------+
            |
            | Model Loading
            |
+-----------v-----------+
|    Model Repository   |
| (models/v1/model.joblib)|
+-----------------------+

```
## For running the overall program run below code 
### uvicorn app.main:app --reload


## API Usage Examples

The API provides endpoints for health checks, predictions, and file uploads.

### Health Check

Check the status of the API and the deployed model version.

```bash
curl -X GET "http://localhost:8000/health"
```

**Example Response:**
```json
{
  "status": "healthy",
  "model_version": "v1"
}
```

### Prediction

Send a POST request with a list of features to get a prediction.

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d 
```

**Example Response:**
```json
{
  "prediction": 0,
  "probability": [0.85, 0.15],
  "status": "success"
}
```

### Upload CSV

Upload a CSV file.

```bash
curl -X POST "http://localhost:8000/upload/csv" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/data.csv"
```

### Upload TXT

Upload a TXT file.

```bash
curl -X POST "http://localhost:8000/upload/txt" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/document.txt"
```

### Upload Image

Upload an image file (JPG, JPEG, PNG, BMP).

```bash
curl -X POST "http://localhost:8000/upload/image" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/image.png"
```

### Upload IPYNB

Upload a Jupyter Notebook file.

```bash
curl -X POST "http://localhost:8000/upload/ipynb" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/notebook.ipynb"
```

## Docker Instructions

To build and run the application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t ml-api .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 ml-api
    ```

The API will be accessible at `http://localhost:8000`.

## Future Improvements

*   **MLflow Integration:** Integrate MLflow for experiment tracking, model registry, and model deployment.
*   **Swagger UI Documentation:** Enable Swagger UI for interactive API documentation.
*   **Environment Variable Configuration:** Externalize configuration using environment variables for better flexibility.
*   **Basic Rate Limiting:** Implement rate limiting to protect the API from abuse.
*   **Model Retraining Pipeline Placeholder:** Add a placeholder for an automated model retraining pipeline.
*   **Monitoring with Prometheus/Grafana:** Integrate Prometheus for metrics collection and Grafana for visualization.

## Monitoring Explanation

The application includes basic logging of requests, predictions, and errors. For production-grade monitoring, it is recommended to integrate with tools like Prometheus and Grafana. Prometheus can scrape custom metrics exposed by the FastAPI application (e.g., request count, prediction latency), and Grafana can be used to create dashboards for real-time visualization and alerting. This setup allows for proactive identification of issues and performance bottlenecks in the ML service.
