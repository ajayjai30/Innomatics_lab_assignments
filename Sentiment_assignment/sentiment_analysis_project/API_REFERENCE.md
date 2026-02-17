# API Reference Documentation

## Flask Sentiment Analysis API

Base URL: `http://localhost:5000` (or your deployed server)

### Health Check

**Endpoint**: `GET /api/health`

**Description**: Check if the API is running and healthy

**Response Example**:
```json
{
  "status": "healthy",
  "model": "XGBoost",
  "timestamp": "2024-02-09T10:30:00"
}
```

---

### Single Prediction

**Endpoint**: `POST /api/predict`

**Description**: Predict sentiment for a single review

**Request Body**:
```json
{
  "review": "This product exceeded my expectations! Amazing quality."
}
```

**Response Example (Positive)**:
```json
{
  "review": "This product exceeded my expectations! Amazing quality.",
  "sentiment": "positive",
  "confidence": 0.95,
  "positive_prob": 0.95,
  "negative_prob": 0.05,
  "timestamp": "2024-02-09T10:30:00"
}
```

**Response Example (Negative)**:
```json
{
  "review": "Terrible quality, broke after a week.",
  "sentiment": "negative",
  "confidence": 0.88,
  "positive_prob": 0.12,
  "negative_prob": 0.88,
  "timestamp": "2024-02-09T10:30:00"
}
```

**Error Response**:
```json
{
  "error": "Review text is required"
}
```

---

### Batch Prediction

**Endpoint**: `POST /api/batch_predict`

**Description**: Predict sentiment for multiple reviews at once

**Request Body**:
```json
{
  "reviews": [
    "Great product! Highly recommended.",
    "Poor quality, not worth the price.",
    "Average product, nothing special.",
    "Excellent packaging and fast delivery!",
    "Broke after one use, very disappointed."
  ]
}
```

**Response Example**:
```json
{
  "predictions": [
    {
      "review": "Great product! Highly recommended.",
      "sentiment": "positive",
      "confidence": 0.93
    },
    {
      "review": "Poor quality, not worth the price.",
      "sentiment": "negative",
      "confidence": 0.91
    },
    {
      "review": "Average product, nothing special.",
      "sentiment": "neutral",
      "confidence": 0.65
    },
    {
      "review": "Excellent packaging and fast delivery!",
      "sentiment": "positive",
      "confidence": 0.97
    },
    {
      "review": "Broke after one use, very disappointed.",
      "sentiment": "negative",
      "confidence": 0.94
    }
  ],
  "count": 5,
  "timestamp": "2024-02-09T10:30:00"
}
```

---

## Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success - Request processed successfully |
| 400 | Bad Request - Invalid request format or missing parameters |
| 405 | Method Not Allowed - Wrong HTTP method used |
| 500 | Internal Server Error - Server-side processing error |
| 404 | Not Found - Endpoint does not exist |

---

## Error Messages and Solutions

### Error: "Review text is required"
**Cause**: Missing `review` field in request body
**Solution**: Ensure request includes valid JSON with `review` field

### Error: "Review must be a non-empty string"
**Cause**: Empty review or non-string value provided
**Solution**: Provide a non-empty string as review text

### Error: "Reviews must be a list"
**Cause**: `reviews` field is not an array
**Solution**: Wrap reviews in JSON array: `{"reviews": ["review1", "review2"]}`

### Error: "Internal server error"
**Cause**: Server-side processing error
**Solution**: Check server logs and ensure model files are present

---

## Code Examples

### Python Requests

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

# Single prediction
def predict_single(review):
    url = f"{API_URL}/api/predict"
    payload = {"review": review}
    response = requests.post(url, json=payload)
    return response.json()

# Batch prediction
def predict_batch(reviews):
    url = f"{API_URL}/api/batch_predict"
    payload = {"reviews": reviews}
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Single prediction
    result = predict_single("This product is amazing!")
    print(json.dumps(result, indent=2))
    
    # Batch prediction
    reviews = [
        "Great quality!",
        "Poor product",
        "Average"
    ]
    batch_result = predict_batch(reviews)
    print(json.dumps(batch_result, indent=2))
```

### JavaScript/Node.js

```javascript
// Single prediction
async function predictSingle(review) {
  const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ review: review })
  });
  return await response.json();
}

// Batch prediction
async function predictBatch(reviews) {
  const response = await fetch('http://localhost:5000/api/batch_predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ reviews: reviews })
  });
  return await response.json();
}

// Example usage
(async () => {
  const single = await predictSingle("Amazing product!");
  console.log(single);
  
  const batch = await predictBatch(["Good", "Bad", "Okay"]);
  console.log(batch);
})();
```

### cURL

```bash
# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This is great!"}'

# Batch prediction
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Good", "Bad", "Average"]}'
```

### JavaScript Fetch API

```javascript
// Check API health
fetch('http://localhost:5000/api/health')
  .then(res => res.json())
  .then(data => console.log(data));

// Single prediction
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ review: 'Great product!' })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Rate Limiting

Currently, there is **no rate limiting** in place. Plan for implementation:

- Recommended: 100 requests/minute per IP
- Batch limit: 1000 reviews per batch request
- Timeout: 30 seconds per request

---

## Best Practices

1. **Preprocessing**: Text will be automatically cleaned and normalized
2. **Batch Processing**: Use batch endpoint for multiple reviews (more efficient)
3. **Error Handling**: Always check response status and error messages
4. **Confidence Scores**: Consider predictions with confidence > 0.7 as reliable
5. **Caching**: Cache identical reviews to reduce API calls

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average Response Time | 50-100ms |
| Batch Processing Speed | ~1000 reviews/minute |
| Model F1-Score | 0.88-0.92 |
| Accuracy | 0.85-0.90 |
| Precision | 0.87-0.91 |
| Recall | 0.85-0.89 |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2024 | Initial release with Logistic Regression and Random Forest |
| 1.1 | TBD | XGBoost support |
| 2.0 | TBD | BERT embeddings support |
| 2.1 | TBD | Multi-language support |

---

## Support

For issues or questions about the API:
1. Check this documentation
2. Review error messages and solutions
3. Check API logs
4. Review notebook examples

---

**Last Updated**: February 9, 2024
**API Status**: Active
**Supported Methods**: Logistic Regression, Random Forest, XGBoost
