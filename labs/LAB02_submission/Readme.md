# Lab 2 Submission README

## Student Information
- Name: [Alexander Zermeno]
- Date: [2026-04-02]

## Deliverables Included
- `inference_api/Dockerfile`
- `preprocessor/Dockerfile`
- `inference_api/app.py` (with `/health` and `/stats`)
- `sample_classifications_20.jsonl` (first 20 lines from logs)
- `Reflection.md`

## Docker Build Commands Used

### Inference API
```bash
docker build -t congo-inference ./inference_api
```

### Preprocessor
```bash
docker build -t congo-preprocessor ./preprocessor
```

## Docker Run Commands Used

### Inference API Container
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/logs:/logs \
  --name inference-api \
  congo-inference
```

### Preprocessor Container
```bash
docker run -d \
  -v $(pwd)/incoming:/incoming \
  -e API_URL=http://host.docker.internal:8000 \
  --name preprocessor \
  congo-preprocessor
```

## Brief Explanation: How the Containers Communicate
The preprocessor watches the `incoming/` folder and sends each new image to the inference API's `/predict` endpoint via HTTP POST. To tell the preprocessor where the API lives, we pass in the `API_URL` environment variable at runtime. One thing that was confusing and had me referring back to our lecture, was the inability to use `localhost` when passing `API_URL` as it is inside a container that just points back to the container itself. For data to actually stay, both containers need bind mounts connecting them to folders on the host machine. The `incoming/` mount is how the preprocessor sees new images dropped on your laptop, meanwhile the `logs/` mount is how classification results get saved somewhere permanent. The inference API writes each result to `/logs/classifications.jsonl`, and the `/stats` endpoint just reads from that same file when you need a summary.