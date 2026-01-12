# RAG Production Architecture on AWS

High-level design for a production-grade RAG system on AWS with chatbot interface, Kubernetes orchestration, and comprehensive monitoring.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Network Architecture](#network-architecture)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Scalability](#scalability)
- [Cost Estimates](#cost-estimates)

---

## Architecture Overview

### High-Level Architecture

```mermaid
graph TB
    subgraph User_Layer["User Layer"]
        User[User Browser]
    end
    
    subgraph AWS_Cloud["AWS Cloud"]
        subgraph Edge_CDN["Edge & CDN"]
            CF[CloudFront CDN]
            S3Web[S3 Static Website<br/>React Frontend]
        end
        
        subgraph Security_Auth["Security & Auth"]
            Cognito[AWS Cognito<br/>User Auth]
            WAF[AWS WAF<br/>Web Application Firewall]
        end
        
        subgraph VPC["VPC - Region: us-east-1"]
            subgraph Public_Subnet["Public Subnet"]
                ALB[Application Load Balancer<br/>Internet-facing]
                NAT[NAT Gateway]
            end
            
            subgraph Private_App["Private Subnet - App Tier"]
                subgraph EKS_Cluster["EKS Cluster"]
                    subgraph API_Pods["API Pods"]
                        API1[FastAPI Pod 1]
                        API2[FastAPI Pod 2]
                        API3[FastAPI Pod 3]
                    end
                    
                    subgraph Worker_Pods["Worker Pods"]
                        Worker1[Document Processor 1]
                        Worker2[Document Processor 2]
                    end
                    
                    subgraph Vector_Pods["Vector DB Pods"]
                        Qdrant1[Qdrant Pod 1]
                        Qdrant2[Qdrant Pod 2]
                        Qdrant3[Qdrant Pod 3]
                    end
                end
                
                ElastiCache[(ElastiCache Redis<br/>Query & Embedding Cache)]
            end
            
            subgraph Private_Data["Private Subnet - Data Tier"]
                S3Docs[(S3 Bucket<br/>Document Storage)]
                EBS[(EBS Volumes<br/>Qdrant Data)]
            end
            
            subgraph Message_Queue["Message Queue"]
                SQS[SQS Queue<br/>Document Processing]
            end
        end
        
        subgraph Monitoring_Stack["Monitoring Stack"]
            subgraph EKS_Mon["EKS Monitoring Pods"]
                Prometheus[Prometheus<br/>Metrics Collection]
                Grafana[Grafana<br/>Visualization]
            end
            
            CloudWatch[CloudWatch<br/>Logs & Metrics]
            XRay[X-Ray<br/>Distributed Tracing]
        end
        
        subgraph External_Services["External Services"]
            OpenAI[OpenAI API<br/>Embeddings & LLM]
            Bedrock[AWS Bedrock<br/>Fallback LLM]
        end
        
        subgraph CICD["CI/CD"]
            GitLab[GitLab CI/CD]
            ECR[Amazon ECR<br/>Container Registry]
        end
    end
    
    User -->|HTTPS| CF
    CF -->|Cache Miss| S3Web
    User -->|API Calls| WAF
    WAF --> ALB
    User -->|Auth| Cognito
    
    ALB --> API1
    ALB --> API2
    ALB --> API3
    
    API1 -->|Check Cache| ElastiCache
    API2 -->|Check Cache| ElastiCache
    API3 -->|Check Cache| ElastiCache
    
    API1 -->|Query Vectors| Qdrant1
    API2 -->|Query Vectors| Qdrant2
    API3 -->|Query Vectors| Qdrant3
    
    API1 -->|Get Embeddings/LLM| OpenAI
    API2 -->|Get Embeddings/LLM| OpenAI
    API3 -->|Get Embeddings/LLM| OpenAI
    
    API1 -->|Fallback LLM| Bedrock
    API2 -->|Fallback LLM| Bedrock
    API3 -->|Fallback LLM| Bedrock
    
    API1 -->|Upload Docs| S3Docs
    API2 -->|Upload Docs| S3Docs
    API3 -->|Upload Docs| S3Docs
    
    S3Docs -->|Trigger| SQS
    SQS --> Worker1
    SQS --> Worker2
    
    Worker1 -->|Store Vectors| Qdrant1
    Worker1 -->|Store Vectors| Qdrant2
    Worker1 -->|Store Vectors| Qdrant3
    Worker2 -->|Store Vectors| Qdrant1
    Worker2 -->|Store Vectors| Qdrant2
    Worker2 -->|Store Vectors| Qdrant3
    
    Worker1 -->|Generate Embeddings| OpenAI
    Worker2 -->|Generate Embeddings| OpenAI
    Worker1 -->|Cache| ElastiCache
    Worker2 -->|Cache| ElastiCache
    
    Qdrant1 -->|Persist| EBS
    Qdrant2 -->|Persist| EBS
    Qdrant3 -->|Persist| EBS
    
    API1 -->|Metrics| Prometheus
    API2 -->|Metrics| Prometheus
    API3 -->|Metrics| Prometheus
    Worker1 -->|Metrics| Prometheus
    Worker2 -->|Metrics| Prometheus
    Qdrant1 -->|Metrics| Prometheus
    Qdrant2 -->|Metrics| Prometheus
    Qdrant3 -->|Metrics| Prometheus
    
    Prometheus --> Grafana
    
    API1 -->|Logs| CloudWatch
    API2 -->|Logs| CloudWatch
    API3 -->|Logs| CloudWatch
    Worker1 -->|Logs| CloudWatch
    Worker2 -->|Logs| CloudWatch
    
    API1 -->|Traces| XRay
    API2 -->|Traces| XRay
    API3 -->|Traces| XRay
    
    GitLab -->|Build & Push| ECR
    ECR -->|Deploy| API1
    ECR -->|Deploy| API2
    ECR -->|Deploy| API3
    ECR -->|Deploy| Worker1
    ECR -->|Deploy| Worker2
    ECR -->|Deploy| Qdrant1
    ECR -->|Deploy| Qdrant2
    ECR -->|Deploy| Qdrant3
    
    style User fill:#e1f5ff
    style CF fill:#ff9900
    style S3Web fill:#569a31
    style Cognito fill:#dd344c
    style WAF fill:#dd344c
    style ALB fill:#8c4fff
    style EKS_Cluster fill:#326ce5
    style ElastiCache fill:#c925d1
    style S3Docs fill:#569a31
    style SQS fill:#ff4f8b
    style Prometheus fill:#e6522c
    style Grafana fill:#f46800
    style CloudWatch fill:#ff9900
    style OpenAI fill:#10a37f
    style Bedrock fill:#ff9900
```

---

## Network Architecture

### VPC & Subnet Design

```mermaid
graph TB
    subgraph "VPC: 10.0.0.0/16"
        subgraph "Availability Zone 1"
            subgraph "Public Subnet 1a<br/>10.0.1.0/24"
                ALB1[ALB]
                NAT1[NAT Gateway]
            end
            
            subgraph "Private App Subnet 1a<br/>10.0.11.0/24"
                EKS1[EKS Worker Nodes<br/>API Pods<br/>Worker Pods<br/>Qdrant Pods]
                Redis1[ElastiCache<br/>Primary Node]
            end
            
            subgraph "Private Data Subnet 1a<br/>10.0.21.0/24"
                S3EP1[S3 VPC Endpoint]
            end
        end
        
        subgraph "Availability Zone 2"
            subgraph "Public Subnet 1b<br/>10.0.2.0/24"
                ALB2[ALB]
                NAT2[NAT Gateway]
            end
            
            subgraph "Private App Subnet 1b<br/>10.0.12.0/24"
                EKS2[EKS Worker Nodes<br/>API Pods<br/>Worker Pods<br/>Qdrant Pods]
                Redis2[ElastiCache<br/>Replica Node]
            end
            
            subgraph "Private Data Subnet 1b<br/>10.0.22.0/24"
                S3EP2[S3 VPC Endpoint]
            end
        end
        
        subgraph "Availability Zone 3"
            subgraph "Public Subnet 1c<br/>10.0.3.0/24"
                ALB3[ALB]
                NAT3[NAT Gateway]
            end
            
            subgraph "Private App Subnet 1c<br/>10.0.13.0/24"
                EKS3[EKS Worker Nodes<br/>API Pods<br/>Worker Pods<br/>Qdrant Pods]
                Redis3[ElastiCache<br/>Replica Node]
            end
            
            subgraph "Private Data Subnet 1c<br/>10.0.23.0/24"
                S3EP3[S3 VPC Endpoint]
            end
        end
        
        IGW[Internet Gateway]
        
        IGW --- ALB1
        IGW --- ALB2
        IGW --- ALB3
        
        NAT1 --- IGW
        NAT2 --- IGW
        NAT3 --- IGW
        
        EKS1 --- NAT1
        EKS2 --- NAT2
        EKS3 --- NAT3
        
        EKS1 -.->|Private Link| S3EP1
        EKS2 -.->|Private Link| S3EP2
        EKS3 -.->|Private Link| S3EP3
    end
    
    Internet[Internet] --- IGW
    
    style IGW fill:#8c4fff
    style NAT1 fill:#8c4fff
    style NAT2 fill:#8c4fff
    style NAT3 fill:#8c4fff
    style EKS1 fill:#326ce5
    style EKS2 fill:#326ce5
    style EKS3 fill:#326ce5
    style Redis1 fill:#c925d1
    style Redis2 fill:#c925d1
    style Redis3 fill:#c925d1
```

---

## Component Details

### EKS Cluster Architecture

```mermaid
graph TB
    subgraph "EKS Control Plane - Managed by AWS"
        EKSCP[EKS API Server<br/>etcd<br/>Scheduler<br/>Controller Manager]
    end
    
    subgraph "Worker Node Group 1 - API Services<br/>Instance: m5.xlarge"
        WN1[Worker Node 1]
        WN2[Worker Node 2]
        WN3[Worker Node 3]
        
        subgraph "API Pods on Node 1"
            API_P1[FastAPI Pod<br/>Resources: 2 CPU, 4GB RAM]
            API_P2[FastAPI Pod<br/>Resources: 2 CPU, 4GB RAM]
        end
        
        subgraph "Monitoring Pods"
            PROM[Prometheus Pod]
            GRAF[Grafana Pod]
        end
    end
    
    subgraph "Worker Node Group 2 - Processing<br/>Instance: c5.2xlarge"
        WN4[Worker Node 4]
        WN5[Worker Node 5]
        
        subgraph "Worker Pods on Node 4"
            WORK_P1[Document Processor<br/>Resources: 4 CPU, 8GB RAM]
            WORK_P2[Document Processor<br/>Resources: 4 CPU, 8GB RAM]
        end
    end
    
    subgraph "Worker Node Group 3 - Vector DB<br/>Instance: r5.xlarge Memory Optimized"
        WN6[Worker Node 6]
        WN7[Worker Node 7]
        WN8[Worker Node 8]
        
        subgraph "Qdrant Pods"
            Q1[Qdrant Pod 1<br/>Resources: 4 CPU, 16GB RAM<br/>100GB EBS gp3]
            Q2[Qdrant Pod 2<br/>Resources: 4 CPU, 16GB RAM<br/>100GB EBS gp3]
            Q3[Qdrant Pod 3<br/>Resources: 4 CPU, 16GB RAM<br/>100GB EBS gp3]
        end
    end
    
    EKSCP --> WN1 & WN2 & WN3 & WN4 & WN5 & WN6 & WN7 & WN8
    
    ALB_K8S[AWS Load Balancer Controller] --> API_P1 & API_P2
    
    HPA[Horizontal Pod Autoscaler] -.->|Scale| API_P1 & API_P2 & WORK_P1 & WORK_P2
    
    CSI[EBS CSI Driver] --> Q1 & Q2 & Q3
    
    style EKSCP fill:#326ce5
    style WN1 fill:#6ca0dc
    style WN2 fill:#6ca0dc
    style WN3 fill:#6ca0dc
    style WN4 fill:#6ca0dc
    style WN5 fill:#6ca0dc
    style WN6 fill:#6ca0dc
    style WN7 fill:#6ca0dc
    style WN8 fill:#6ca0dc
```

---

## Data Flow

### Query Flow (User Question)

```mermaid
sequenceDiagram
    participant U as User Browser
    participant CF as CloudFront
    participant FE as React Frontend
    participant ALB as ALB
    participant API as FastAPI Pod
    participant Cache as ElastiCache
    participant Q as Qdrant
    participant OpenAI as OpenAI API
    participant CW as CloudWatch
    
    U->>CF: GET / (Load App)
    CF->>FE: Serve React App
    FE->>U: Display Chat UI
    
    U->>ALB: POST /api/ask {"query": "What is...?"}
    Note over ALB: JWT Token Validation
    ALB->>API: Forward Request
    
    API->>CW: Log Request
    API->>Cache: Check Query Cache
    
    alt Cache Hit
        Cache-->>API: Return Cached Answer
        API-->>ALB: 200 OK + Answer
        ALB-->>U: Response
    else Cache Miss
        API->>Cache: Check Embedding Cache
        alt Embedding Cached
            Cache-->>API: Return Embedding
        else Generate Embedding
            API->>OpenAI: POST /embeddings
            OpenAI-->>API: Return Embedding
            API->>Cache: Store Embedding (TTL: 24h)
        end
        
        API->>Q: Vector Search (query_embedding)
        Q-->>API: Top 3 Relevant Chunks
        
        API->>OpenAI: POST /chat/completions<br/>(context + query)
        OpenAI-->>API: Generated Answer
        
        API->>Cache: Store Answer (TTL: 1h)
        API->>CW: Log Success + Latency
        API-->>ALB: 200 OK + Answer
        ALB-->>U: Response
    end
```

### Document Ingestion Flow

```mermaid
sequenceDiagram
    participant U as User Browser
    participant API as FastAPI Pod
    participant S3 as S3 Bucket
    participant SQS as SQS Queue
    participant W as Worker Pod
    participant OpenAI as OpenAI API
    participant Q as Qdrant
    participant Cache as ElastiCache
    participant CW as CloudWatch
    
    U->>API: POST /api/upload (PDF file)
    API->>S3: Upload to s3://docs/{uuid}.pdf
    S3-->>API: Upload Success
    
    API->>SQS: Send Message<br/>{doc_id, s3_key, user_id}
    API-->>U: 202 Accepted {job_id}
    
    Note over W: Polling SQS
    SQS->>W: Receive Message
    W->>CW: Log: Processing Started
    
    W->>S3: Download Document
    S3-->>W: PDF Content
    
    W->>W: Extract Text (PyPDF)
    W->>W: Chunk Text (500 words, 50 overlap)
    Note over W: Generated 47 chunks
    
    loop For Each Chunk
        W->>OpenAI: POST /embeddings
        OpenAI-->>W: Embedding Vector
        W->>W: Add to Batch
    end
    
    W->>Q: Batch Insert Vectors<br/>(chunks + embeddings + metadata)
    Q-->>W: Insert Success
    
    W->>Cache: Invalidate Related Queries
    W->>SQS: Delete Message
    W->>CW: Log: Processing Complete
    
    Note over API: Webhook/SSE to notify user
    API->>U: Document Ready Notification
```

---

## Monitoring & Observability

### Monitoring Stack

```mermaid
graph TB
    subgraph "Application Layer"
        API[FastAPI Pods]
        Worker[Worker Pods]
        Qdrant[Qdrant Pods]
    end
    
    subgraph "Metrics Collection"
        API -->|Expose :9090/metrics| PM1[Prometheus]
        Worker -->|Expose :9090/metrics| PM1
        Qdrant -->|Expose :9090/metrics| PM1
        
        PM1[Prometheus<br/>Scrape Interval: 15s<br/>Retention: 15d]
    end
    
    subgraph "Visualization"
        PM1 --> Grafana[Grafana<br/>Dashboards]
        
        Grafana --> D1[Dashboard: API Performance<br/>- Request Rate<br/>- Latency p50/p95/p99<br/>- Error Rate<br/>- Cache Hit Rate]
        
        Grafana --> D2[Dashboard: Vector DB<br/>- Query Performance<br/>- Storage Usage<br/>- Index Size<br/>- Search Latency]
        
        Grafana --> D3[Dashboard: Cost Tracking<br/>- OpenAI API Costs<br/>- Token Usage<br/>- Cache Savings]
        
        Grafana --> D4[Dashboard: Business Metrics<br/>- Active Users<br/>- Queries per User<br/>- Popular Questions<br/>- User Satisfaction]
    end
    
    subgraph "Logging"
        API -->|stdout/stderr| Fluent[Fluent Bit DaemonSet]
        Worker -->|stdout/stderr| Fluent
        Qdrant -->|stdout/stderr| Fluent
        
        Fluent --> CW[CloudWatch Logs<br/>Log Groups]
        CW --> CWI[CloudWatch Insights<br/>Query & Analysis]
    end
    
    subgraph "Tracing"
        API -->|OpenTelemetry| XRay[AWS X-Ray]
        XRay --> XMap[Service Map]
        XRay --> XTrace[Trace Analysis]
    end
    
    subgraph "Alerting"
        PM1 --> AM[AlertManager]
        AM -->|Critical| PD[PagerDuty]
        AM -->|Warning| Slack[Slack Channel]
        AM -->|Info| Email[Email]
        
        CW --> CWA[CloudWatch Alarms]
        CWA -->|High Severity| SNS[SNS Topic]
        SNS --> PD
    end
    
    style PM1 fill:#e6522c
    style Grafana fill:#f46800
    style CW fill:#ff9900
    style XRay fill:#ff9900
    style AM fill:#e6522c
```

### Key Metrics to Monitor

```mermaid
graph LR
    subgraph "API Metrics"
        M1[Request Rate<br/>requests/sec]
        M2[Latency<br/>p50, p95, p99]
        M3[Error Rate<br/>4xx, 5xx %]
        M4[Concurrent Users]
    end
    
    subgraph "Cache Metrics"
        M5[Hit Rate %]
        M6[Miss Rate %]
        M7[Eviction Rate]
        M8[Memory Usage]
    end
    
    subgraph "Vector DB Metrics"
        M9[Search Latency<br/>ms]
        M10[Index Size<br/>GB]
        M11[Vector Count]
        M12[Storage I/O]
    end
    
    subgraph "Cost Metrics"
        M13[OpenAI API Cost<br/>$/day]
        M14[Token Usage<br/>tokens/day]
        M15[Cache Savings<br/>$]
        M16[Infrastructure<br/>$/month]
    end
    
    subgraph "Business Metrics"
        M17[Daily Active Users]
        M18[Avg Queries/User]
        M19[Query Success Rate]
        M20[User Satisfaction<br/>Thumbs up/down]
    end
```

---

## Security

### Security Layers

```mermaid
graph TB
    subgraph "Perimeter Security"
        WAF[AWS WAF<br/>- SQL Injection Protection<br/>- XSS Protection<br/>- Rate Limiting<br/>- Geo Blocking]
        
        Shield[AWS Shield Standard<br/>DDoS Protection]
    end
    
    subgraph "Authentication & Authorization"
        Cognito[AWS Cognito<br/>- User Pools<br/>- OAuth 2.0<br/>- MFA Support]
        
        IAM[IAM Roles<br/>- EKS Pod IAM Roles<br/>- Service Accounts<br/>- Least Privilege]
    end
    
    subgraph "Network Security"
        SG[Security Groups<br/>- ALB: 443 from 0.0.0.0/0<br/>- EKS: 443 from ALB<br/>- Qdrant: 6333 from API Pods<br/>- Redis: 6379 from API/Workers]
        
        NACL[Network ACLs<br/>Subnet-level Rules]
        
        PL[VPC PrivateLink<br/>- S3 Endpoint<br/>- ECR Endpoint<br/>- CloudWatch Endpoint]
    end
    
    subgraph "Data Security"
        KMS[AWS KMS<br/>- Encrypt EBS Volumes<br/>- Encrypt S3 Objects<br/>- Encrypt Secrets]
        
        SM[AWS Secrets Manager<br/>- OpenAI API Key<br/>- Database Credentials<br/>- JWT Secret]
        
        TLS[TLS 1.3<br/>- ALB Certificate<br/>- Internal Service Mesh]
    end
    
    subgraph "Application Security"
        RBAC[Kubernetes RBAC<br/>- Role-based Access<br/>- ServiceAccount Tokens]
        
        PSP[Pod Security Standards<br/>- Restricted<br/>- No Privileged Containers<br/>- Read-only Root FS]
        
        NP[Network Policies<br/>- Deny All by Default<br/>- Explicit Allow Rules]
    end
    
    subgraph "Compliance & Audit"
        CT[CloudTrail<br/>API Call Logging]
        
        Config[AWS Config<br/>Compliance Monitoring]
        
        GRC[GuardDuty<br/>Threat Detection]
    end
    
    Internet[Internet] --> Shield
    Shield --> WAF
    WAF --> ALB[ALB]
    
    User[User] --> Cognito
    Cognito -.->|JWT Token| ALB
    
    ALB --> SG
    SG --> EKS[EKS Pods]
    
    EKS --> IAM
    EKS --> RBAC
    EKS --> PSP
    EKS --> NP
    
    EKS --> KMS
    EKS --> SM
    EKS -.->|Private| PL
    
    CT & Config & GRC -.->|Monitoring| SecurityTeam[Security Team]
    
    style WAF fill:#dd344c
    style Shield fill:#dd344c
    style Cognito fill:#dd344c
    style KMS fill:#dd344c
    style SM fill:#dd344c
```

### Data Flow Security

```mermaid
graph LR
    U[User] -->|HTTPS TLS 1.3| CF[CloudFront]
    CF -->|HTTPS TLS 1.3| ALB[ALB]
    ALB -->|HTTP + JWT| API[API Pod]
    API -->|mTLS| Qdrant[Qdrant Pod]
    API -->|TLS + Redis AUTH| Cache[ElastiCache]
    API -->|HTTPS + API Key| OpenAI[OpenAI API]
    API -->|Encrypted| S3[S3 SSE-KMS]
    
    style U fill:#e1f5ff
    style CF fill:#ff9900
    style ALB fill:#8c4fff
    style API fill:#326ce5
    style S3 fill:#569a31
```

---

## Scalability

### Auto-Scaling Strategy

```mermaid
graph TB
    subgraph "Horizontal Pod Autoscaler"
        HPA[HPA Configuration]
        
        HPA --> API_HPA[API Pods<br/>Min: 3, Max: 20<br/>Target CPU: 70%<br/>Target Memory: 80%]
        
        HPA --> Worker_HPA[Worker Pods<br/>Min: 2, Max: 10<br/>Target CPU: 80%<br/>Custom: Queue Depth]
        
        HPA --> Qdrant_HPA[Qdrant Pods<br/>Min: 3, Max: 6<br/>Target Memory: 85%]
    end
    
    subgraph "Cluster Autoscaler"
        CA[Cluster Autoscaler]
        
        CA --> NG1[Node Group 1: API<br/>Min: 3, Max: 10<br/>m5.xlarge]
        
        CA --> NG2[Node Group 2: Workers<br/>Min: 2, Max: 8<br/>c5.2xlarge]
        
        CA --> NG3[Node Group 3: Qdrant<br/>Min: 3, Max: 6<br/>r5.xlarge]
    end
    
    subgraph "ElastiCache Scaling"
        Redis[ElastiCache Redis<br/>Node Type: cache.r6g.large<br/>Replicas: 2-5]
        
        Redis --> RA[Read Replicas<br/>Auto-Scale Based on CPU]
    end
    
    subgraph "Database Scaling"
        Qdrant_Scale[Qdrant Horizontal Scaling<br/>Sharding Strategy:<br/>- Collection Sharding<br/>- Replication Factor: 2]
    end
    
    M[Metrics Server] -->|CPU/Memory| HPA
    M -->|Node Pressure| CA
    
    CW[CloudWatch] -->|Custom Metrics| HPA
    CW -->|Redis Metrics| Redis
```

### Traffic Patterns & Capacity

```mermaid
graph LR
    subgraph "Normal Load"
        N1[100 req/sec<br/>~8,640,000 req/day]
        N1 --> N2[3 API Pods<br/>3 Worker Pods<br/>3 Qdrant Pods]
    end
    
    subgraph "Peak Load"
        P1[500 req/sec<br/>Business Hours]
        P1 --> P2[10 API Pods<br/>6 Worker Pods<br/>4 Qdrant Pods]
    end
    
    subgraph "Burst Load"
        B1[1000 req/sec<br/>Marketing Campaign]
        B1 --> B2[20 API Pods<br/>10 Worker Pods<br/>6 Qdrant Pods]
    end
    
    N2 -->|Auto-scale| P2
    P2 -->|Auto-scale| B2
    B2 -->|Scale Down| N2
```

---

## Cost Estimates

### Monthly Infrastructure Costs (Production)

```mermaid
graph TB
    subgraph "Compute Costs"
        EKS[EKS Control Plane<br/>$73/month<br/>per cluster]
        
        Nodes[EC2 Worker Nodes<br/>8 nodes average<br/>~$800/month]
        
        NAT[NAT Gateway<br/>3 AZs × $32<br/>~$96/month]
    end
    
    subgraph "Storage & Database"
        EBS[EBS Volumes<br/>300GB gp3<br/>~$24/month]
        
        S3[S3 Storage<br/>500GB Standard<br/>~$12/month]
        
        Redis[ElastiCache Redis<br/>cache.r6g.large + replicas<br/>~$200/month]
    end
    
    subgraph "Networking"
        ALB_Cost[Application Load Balancer<br/>~$25/month + $0.008/LCU-hour<br/>~$50/month total]
        
        DataTransfer[Data Transfer Out<br/>~$90/GB<br/>~$45/month]
    end
    
    subgraph "Monitoring & Logging"
        CW[CloudWatch<br/>Logs + Metrics<br/>~$30/month]
        
        XRay_Cost[X-Ray<br/>~$5/month]
    end
    
    subgraph "API Costs"
        OpenAI_Cost[OpenAI API<br/>Depends on Usage<br/>~$200-500/month]
        
        Bedrock_Cost[AWS Bedrock<br/>Fallback Only<br/>~$50/month]
    end
    
    Total[Total Monthly Cost<br/>Infrastructure: ~$1,400<br/>APIs: ~$250-550<br/>TOTAL: ~$1,650-1,950]
    
    EKS & Nodes & NAT --> Total
    EBS & S3 & Redis --> Total
    ALB_Cost & DataTransfer --> Total
    CW & XRay_Cost --> Total
    OpenAI_Cost & Bedrock_Cost --> Total
    
    style Total fill:#ff9900
```

### Cost Optimization Strategies

```mermaid
graph TB
    subgraph "Compute Optimization"
        Spot[Use Spot Instances<br/>for Worker Nodes<br/>Save: 60-70%]
        
        Savings[Savings Plans<br/>1-year commitment<br/>Save: 30-40%]
        
        RightSize[Right-size Instances<br/>Based on Metrics<br/>Save: 20-30%]
    end
    
    subgraph "Storage Optimization"
        S3_Lifecycle[S3 Lifecycle Policies<br/>Move to Glacier after 90d<br/>Save: 70%]
        
        EBS_GP3[Use gp3 vs gp2<br/>Same performance<br/>Save: 20%]
    end
    
    subgraph "Caching Strategy"
        CacheMore[Aggressive Caching<br/>Reduce API Calls<br/>Save: 40-60% on API costs]
    end
    
    subgraph "API Cost Reduction"
        BatchEmbed[Batch Embeddings<br/>Reduce API overhead<br/>Save: 10-15%]
        
        LocalEmbed[Consider Local Models<br/>for Dev/Test<br/>Save: 100% in non-prod]
    end
    
    Spot & Savings & RightSize --> Savings1[~40% Compute Savings]
    S3_Lifecycle & EBS_GP3 --> Savings2[~30% Storage Savings]
    CacheMore & BatchEmbed --> Savings3[~50% API Cost Savings]
    
    Savings1 & Savings2 & Savings3 --> Total[Potential Monthly Savings<br/>~$600-800/month]
    
    style Total fill:#569a31
```

---

## Deployment Strategy

### CI/CD Pipeline

```mermaid
graph LR
    Dev[Developer] -->|git push| GitLab[GitLab Repository]
    
    GitLab -->|Trigger| CI[GitLab CI Pipeline]
    
    subgraph "CI Pipeline"
        CI --> Build[1. Build<br/>Docker Image]
        Build --> Test[2. Run Tests<br/>Unit + Integration]
        Test --> Scan[3. Security Scan<br/>Trivy + Snyk]
        Scan --> Push[4. Push to ECR<br/>Tag: commit-sha]
    end
    
    Push --> CD[GitLab CD Pipeline]
    
    subgraph "CD Pipeline - Staging"
        CD --> Deploy_Stg[1. Deploy to Staging<br/>EKS Cluster]
        Deploy_Stg --> Test_Stg[2. Smoke Tests]
        Test_Stg --> Approve[3. Manual Approval]
    end
    
    Approve --> Prod[CD Pipeline - Production]
    
    subgraph "CD Pipeline - Production"
        Prod --> Deploy_Prod[1. Blue/Green Deploy<br/>EKS Cluster]
        Deploy_Prod --> Health[2. Health Checks]
        Health --> Switch[3. Switch Traffic]
        Switch --> Monitor[4. Monitor 15min]
        Monitor -->|Success| Complete[Deployment Complete]
        Monitor -->|Failure| Rollback[Automatic Rollback]
    end
    
    style GitLab fill:#fc6d26
    style ECR fill:#ff9900
    style Complete fill:#569a31
    style Rollback fill:#dd344c
```

---

## Disaster Recovery

### Backup Strategy

```mermaid
graph TB
    subgraph "Data Backup"
        S3[S3 Documents<br/>Versioning Enabled<br/>Cross-Region Replication<br/>to us-west-2]
        
        EBS[EBS Snapshots<br/>Daily Automated Snapshots<br/>Retention: 30 days<br/>Cross-Region Copy]
        
        Qdrant_Backup[Qdrant Backups<br/>Daily Full Backup to S3<br/>Incremental Snapshots<br/>Retention: 90 days]
    end
    
    subgraph "Configuration Backup"
        GitLab_IaC[Infrastructure as Code<br/>Terraform State in S3<br/>Kubernetes Manifests in Git]
        
        Secrets[Secrets Backup<br/>AWS Secrets Manager<br/>Automatic Replication<br/>to Secondary Region]
    end
    
    subgraph "Recovery Time Objectives"
        RTO[RTO: 2 hours<br/>Time to restore service]
        
        RPO[RPO: 1 hour<br/>Maximum data loss]
    end
    
    DR[Disaster Recovery Plan<br/>Documented Runbook<br/>Quarterly DR Drills]
    
    S3 & EBS & Qdrant_Backup --> DR
    GitLab_IaC & Secrets --> DR
    DR --> RTO & RPO
```

---

## Summary

### Architecture Highlights

| Component | Technology | Purpose | Redundancy |
|-----------|-----------|---------|------------|
| **Frontend** | React on S3 + CloudFront | User Interface | Multi-AZ CDN |
| **API Gateway** | Application Load Balancer | Traffic Distribution | Multi-AZ |
| **API Layer** | FastAPI on EKS | Business Logic | 3-20 pods (auto-scale) |
| **Vector Database** | Qdrant on EKS | Semantic Search | 3-6 pods (replicated) |
| **Cache** | ElastiCache Redis | Query & Embedding Cache | Primary + 2 replicas |
| **Message Queue** | SQS | Async Processing | Managed, Multi-AZ |
| **Storage** | S3 | Document Storage | 99.999999999% durability |
| **Monitoring** | Prometheus + Grafana | Metrics & Alerts | Persistent storage |
| **Logging** | CloudWatch Logs | Centralized Logging | Managed |
| **Tracing** | AWS X-Ray | Distributed Tracing | Managed |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **API Latency (p95)** | < 2 seconds | Including LLM call |
| **API Latency (p99)** | < 5 seconds | Cold cache scenario |
| **Cache Hit Rate** | > 60% | For similar queries |
| **Vector Search** | < 100ms | Within Qdrant |
| **Availability** | 99.9% | ~8.76 hours downtime/year |
| **Document Processing** | < 5 minutes | For 100-page PDF |

### Scalability Targets

| Metric | Current | Scale Target |
|--------|---------|--------------|
| **Concurrent Users** | 100-500 | 5,000+ |
| **Requests/Second** | 100 | 1,000+ |
| **Documents** | 10,000 | 1,000,000+ |
| **Vector Dimension** | 1,536 | 3,072 (if needed) |
| **Storage** | 100GB | 10TB+ |

---

## Next Steps for Implementation

1. **Phase 1: Foundation (Week 1-2)**
   - Set up VPC, subnets, security groups
   - Deploy EKS cluster
   - Set up ECR and GitLab CI/CD

2. **Phase 2: Core Services (Week 3-4)**
   - Deploy FastAPI application
   - Deploy Qdrant cluster
   - Set up ElastiCache
   - Configure S3 and SQS

3. **Phase 3: Monitoring (Week 5)**
   - Deploy Prometheus and Grafana
   - Configure CloudWatch integration
   - Set up alerting rules

4. **Phase 4: Frontend (Week 6)**
   - Deploy React application
   - Configure CloudFront
   - Set up Cognito authentication

5. **Phase 5: Testing & Optimization (Week 7-8)**
   - Load testing
   - Performance tuning
   - Cost optimization
   - Security hardening

6. **Phase 6: Production Readiness (Week 9-10)**
   - DR testing
   - Documentation
   - Team training
   - Go-live checklist

---

## Additional Resources

- **Terraform Modules**: Infrastructure as Code for all components
- **Kubernetes Manifests**: Deployment configs for all services
- **Monitoring Dashboards**: Pre-built Grafana dashboards
- **Runbooks**: Operational procedures for common scenarios
- **API Documentation**: OpenAPI/Swagger specs

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Owner:** Platform Engineering Team