# Production RAG Architecture (AWS)

This document describes the production-grade Retrieval-Augmented Generation (RAG)
architecture deployed on AWS.

---

## High-Level Architecture (AWS Icons)

```mermaid
architecture-beta
    %% =======================
    %% Internet & Edge
    %% =======================
    actor user as "User"

    service cloudfront as "CloudFront" {
        icon: aws:cloudfront
    }

    service s3_frontend as "S3 (React App)" {
        icon: aws:s3
    }

    service cognito as "Cognito" {
        icon: aws:cognito
    }

    user --> cloudfront
    cloudfront --> s3_frontend
    user --> cognito

    %% =======================
    %% VPC
    %% =======================
    group vpc as "VPC" {

        %% ---------- Public Subnet ----------
        group public_subnet as "Public Subnet" {

            service alb as "ALB / Ingress" {
                icon: aws:elastic-load-balancing
            }
        }

        %% ---------- Private Subnet ----------
        group private_subnet as "Private Subnet" {

            service eks_api as "EKS - FastAPI" {
                icon: aws:eks
            }

            service qdrant as "Qdrant (Vector DB)" {
                icon: aws:eks
            }

            service redis as "ElastiCache (Redis)" {
                icon: aws:elasticache
            }

            service ingest_workers as "EKS Ingestion Workers" {
                icon: aws:eks
            }
        }
    }

    cloudfront --> alb
    alb --> eks_api
    cognito --> eks_api

    %% =======================
    %% RAG Flow
    %% =======================
    eks_api --> redis : "Cache lookup"
    eks_api --> qdrant : "Vector search"

    service openai as "OpenAI API" {
        icon: aws:cloud
    }

    service bedrock as "Bedrock" {
        icon: aws:bedrock
    }

    eks_api --> openai : "Primary LLM"
    eks_api --> bedrock : "Fallback LLM"

    %% =======================
    %% Ingestion Pipeline
    %% =======================
    service s3_docs as "S3 (Documents)" {
        icon: aws:s3
    }

    service lambda as "Lambda" {
        icon: aws:lambda
    }

    service sqs as "SQS" {
        icon: aws:sqs
    }

    s3_docs --> lambda
    lambda --> sqs
    sqs --> ingest_workers
    ingest_workers --> qdrant

    %% =======================
    %% Observability
    %% =======================
    service cloudwatch as "CloudWatch" {
        icon: aws:cloudwatch
    }

    service prometheus as "Prometheus" {
        icon: aws:cloud
    }

    service grafana as "Grafana" {
        icon: aws:cloud
    }

    eks_api --> cloudwatch
    lambda --> cloudwatch
    alb --> cloudwatch

    eks_api --> prometheus
    prometheus --> grafana

    %% =======================
    %% CI/CD
    %% =======================
    service gitlab as "GitLab CI" {
        icon: aws:cloud
    }

    gitlab --> eks_api
    gitlab --> s3_frontend
