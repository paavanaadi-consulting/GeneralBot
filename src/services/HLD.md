flowchart LR
    %% ===== Frontend =====
    User((User))
    CF[CloudFront]
    S3FE[S3 - React App]

    User --> CF --> S3FE

    %% ===== Auth =====
    Cognito[AWS Cognito]
    User --> Cognito

    %% ===== API Layer =====
    CF --> ALB
    ALB[ALB / Ingress]
    EKSAPI[EKS - FastAPI Service]

    Cognito -->|JWT| EKSAPI

    %% ===== RAG Core =====
    subgraph EKS["EKS Cluster"]
        EKSAPI --> Retriever[RAG Retriever]
        Retriever --> Qdrant[(Qdrant Vector DB)]
        Retriever --> LLMRouter[LLM Router]
        LLMRouter --> OpenAI[OpenAI API]
        LLMRouter --> Bedrock[AWS Bedrock]
    end

    %% ===== Ingestion Pipeline =====
    S3Docs[S3 - Raw Documents]
    LambdaIngest[Lambda - Preprocess]
    SQS[SQS Queue]
    EKSWorkers[EKS - Ingestion Workers]

    S3Docs --> LambdaIngest --> SQS --> EKSWorkers
    EKSWorkers --> Qdrant

    %% ===== Observability =====
    Prometheus[Prometheus]
    Grafana[Grafana]
    CloudWatch[CloudWatch]

    EKS --> Prometheus --> Grafana
    LambdaIngest --> CloudWatch
    ALB --> CloudWatch
    EKS --> CloudWatch

    %% ===== CI/CD =====
    GitLab[GitLab CI]
    GitLab -->|Build & Deploy| EKS
    GitLab -->|Sync Frontend| S3FE
