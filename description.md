 Assignment: Multi-Camera Computer Vision Pipeline Design

Goal
Design a scalable computer vision system that:

    Streams data from up to 50 RTSP cameras
    Performs real-time image preprocessing
    Sends frames to a separate inference service
    Runs a multi-stage detection pipeline
    Aggregates inference results into a final hypothesis

Preferred language: Python
You may use open-source libraries and frameworks of your choice.
System Requirements

Multi-Camera Streaming Layer
Design a module that:

    Connects to up to 50 RTSP streams simultaneously
    Handles:
        Stream failures and reconnections
        Frame drops
        Back-pressure
    Supports configurable:
        FPS per stream
        Resolution
        Buffer size

You may use:

    Nvdia DeepStream (perfered)
    OpenCV
    GStreamer
    FFmpeg
    Async frameworks (asyncio, multiprocessing, threading)
    Message brokers (Kafka, Redis, ZeroMQ, etc.)

Preprocessing Layer
For each incoming frame  perform one of these preprocessing:

    Resize (configurable)
    Noise reduction
    Deblurring (optional)
    Normalization
    Any additional enhancements you consider important)

Design considerations:

    CPU vs GPU trade-offs
    Batch vs per-frame processing
    Throughput optimization

 Inference Service (Separate Microservice)
Design inference as an independent service (not embedded inside the streaming service).
It must:
Step A: Object Detection

    Choose any detector (e.g., YOLO, DETR, Faster R-CNN)
    Output bounding boxes + confidence scores

Step B: Secondary Detection (Cropped Inference)

    Crop detected bounding boxes
    Pass crops to second model (e.g., OCR or classifier)
    Produce structured output

Examples:

    Vehicle → Crop license plate → OCR
    Person → Crop badge → OCR

 Data Integrator Layer
Design a module that:

    Collects inference results
    Associates results across:
        Cameras
        Time
        Object IDs (optional tracking)
    Produces final hypothesis

Examples:

    “Vehicle ABC123 detected at Gate 3”
    “Unauthorized person detected in Zone A”
    “Smoke detected in Yard Section B”

Expected Delivery
System Architecture
Provide:

    High-level architecture diagram
    Component-level breakdown
    Data flow description
    Deployment strategy (Docker, Kubernetes optional)

Technical Design Document (Required)
Explain:

    Concurrency model
    Queueing strategy
    Back-pressure handling
    Failure recovery
    Scaling strategy (horizontal/vertical)
    GPU utilization strategy
    Latency estimates
    Throughput estimates

Code (Prototype Level)
Implement:

    RTSP streaming for multiple cameras (mock cameras acceptable)
    Preprocessing module
    Inference service stub (can simulate inference)
    End-to-end data flow
    Logging & metrics

It does not need to be production-grade, but structure and clarity matter.
performance Considerations
Discuss:

    How to scale from 5 → 50 → 200 cameras
    How to handle GPU bottlenecks
    Model batching strategies
    How to avoid memory leaks
    Monitoring strategy (Prometheus, logging, etc.)

 Bonus (Optional but Strongly Preferred)

    Implement asynchronous pipeline
    Use message broker (Kafka/Redis)
    Deploy using Docker Compose
