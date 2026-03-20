# Lab 2 Submission README

## Student Information
- Name: [Your Name]
- Date: [YYYY-MM-DD]

## Deliverables Included
- `inference_api/Dockerfile`
- `preprocessor/Dockerfile`
- `inference_api/app.py` (with `/health` and `/stats`)
- `sample_classifications_20.jsonl` (first 20 lines from logs)
- `Reflection.md`

## Docker Build Commands Used

### Inference API
```bash
[PUT YOUR DOCKER BUILD COMMAND FOR INFERENCE API IMAGE HERE]
```

### Preprocessor
```bash
[PUT YOUR DOCKER BUILD COMMAND FOR PREPROCESSOR IMAGE HERE]
```

## Docker Run Commands Used

### Inference API Container
```bash
[PUT YOUR DOCKER RUN COMMAND FOR INFERENCE API CONTAINER HERE]
```

### Preprocessor Container
```bash
[PUT YOUR DOCKER RUN COMMAND FOR PREPROCESSOR CONTAINER HERE]
```

## Brief Explanation: How the Containers Communicate
[Write 3-6 sentences here.]

Points to cover:
- Which container calls which endpoint.
- How the preprocessor knows where to find the inference API.
- How images and logs persist using mounted host folders.
- Why `localhost` can be tricky inside containers.

