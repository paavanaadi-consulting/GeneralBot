# RAG MVP Implementation Plan

**Team:** Abhijeet (DevOps/Data Engineering) + Ashoka (Frontend/Backend)  
**Time:** 3-4 hours/week each  
**Budget:** ~$180/month  
**Timeline:** 6 weeks

---

## Technology Stack

- **Cloud:** AWS (us-east-1), Default VPC
- **Orchestration:** EKS (2x t3.medium nodes)
- **IaC:** Terraform (state in S3)
- **Backend:** FastAPI
- **Frontend:** React on S3
- **Vector DB:** Qdrant on EKS
- **Cache:** ElastiCache Redis (cache.t3.micro)
- **AI:** OpenAI (embeddings + GPT-3.5)
- **Monitoring:** Prometheus + Grafana + CloudWatch

---

## Week 0: Setup (2 hours each)

### Abhijeet (AWS + Tools)
- [ ] Create AWS account, set billing alerts ($50, $100, $150, $200)
- [ ] Create IAM admin user, configure AWS CLI
- [ ] Install: terraform, kubectl, helm, eksctl
- [ ] Create S3 bucket for Terraform state
- [ ] Create DynamoDB table for state locking

### Ashoka (AI + Development Tools)
- [ ] Create OpenAI account, get API key, set $50 usage limit
- [ ] Test OpenAI embeddings and chat APIs
- [ ] Install Docker, run Qdrant locally
- [ ] Test Qdrant: create collection, insert/search vectors
- [ ] Install Python, Node.js

### Both
- [ ] Create GitHub repo, share access
- [ ] Create project structure, commit

---

## Week 1: Infrastructure & Backend Foundation (4 hours each)

### Abhijeet (DevOps: EKS + Monitoring)
- [ ] Write Terraform for EKS cluster (default VPC, 2x t3.medium nodes)
- [ ] Apply Terraform, configure kubectl
- [ ] Add Prometheus Helm repo
- [ ] Deploy kube-prometheus-stack (Prometheus + Grafana)
- [ ] Port-forward Grafana, access UI
- [ ] Enable CloudWatch Container Insights
- [ ] Create basic cluster dashboard (CPU, memory, pods)

### Ashoka (DevOps: Vector DB + Backend)
- [ ] Create k8s manifests for Qdrant (deployment, service, PVC 10GB)
- [ ] Get cluster access from Abhijeet
- [ ] Deploy Qdrant to EKS
- [ ] Test Qdrant connection, create test collection
- [ ] Create FastAPI project structure
- [ ] Create health endpoint, OpenAI wrapper, Qdrant wrapper
- [ ] Test locally

---

## Week 2: Storage, Cache & Document Processing (4 hours each)

### Abhijeet (Backend: Document Upload & Processing)
- [ ] Implement upload endpoint (POST /upload)
- [ ] Add S3 upload, text extraction, chunking (500 words, 50 overlap)
- [ ] Implement embedding generation (batch mode)
- [ ] Store vectors in Qdrant with metadata
- [ ] Test end-to-end: upload → chunk → embed → store

### Ashoka (DevOps: S3, Redis & Monitoring)
- [ ] Add Terraform for S3 bucket (versioning, CORS)
- [ ] Add Terraform for ElastiCache Redis (cache.t3.micro)
- [ ] Apply Terraform
- [ ] Get Redis endpoint, test from EKS pod
- [ ] Configure Prometheus to scrape Qdrant
- [ ] Create Qdrant dashboard in Grafana
- [ ] Set up CloudWatch log groups, install Fluent Bit

---

## Week 3: Query API & Deployment (4 hours each)

### Abhijeet (Backend: Query Endpoint & Caching)
- [ ] Implement query endpoint (POST /ask)
- [ ] Generate query embedding, search Qdrant (top 3)
- [ ] Build context, call OpenAI for answer
- [ ] Add Redis client
- [ ] Implement query cache (1hr TTL) and embedding cache (24hr TTL)
- [ ] Test cache effectiveness

### Ashoka (DevOps: Container & Kubernetes Deployment)
- [ ] Write Dockerfile for FastAPI
- [ ] Build image, create ECR repo, push to ECR
- [ ] Write k8s manifests (deployment, service, configmap, secret)
- [ ] Deploy to EKS (2 replicas, LoadBalancer service)
- [ ] Get external URL, test from internet
- [ ] Add Prometheus metrics endpoint to FastAPI
- [ ] Create API dashboard (requests, latency, errors)

---

## Week 4: Frontend Development (3-4 hours each)

### Abhijeet (Frontend: React UI Components)
- [ ] Create React app with create-react-app
- [ ] Create components: Chat, Upload, Message
- [ ] Create API service wrapper (axios)
- [ ] Build Chat UI (message list, input, send button)
- [ ] Build Upload UI (file picker, upload button)
- [ ] Connect to API endpoints
- [ ] Test locally, fix CORS issues

### Ashoka (DevOps: Frontend Infrastructure & Monitoring)
- [ ] Verify LoadBalancer, get external URL
- [ ] Test API endpoints, configure CORS
- [ ] Add Terraform for S3 frontend bucket
- [ ] Enable static website hosting, public read policy
- [ ] Apply Terraform, test with HTML file
- [ ] Create business metrics dashboard in Grafana

---

## Week 5: Deployment & Testing (3-4 hours each)

### Abhijeet (Data Engineering: Testing & Validation)
- [ ] Upload 5-10 test PDFs
- [ ] Test queries (simple, complex, edge cases)
- [ ] Verify cache working (check Redis, logs)
- [ ] Check query latency, cache hit rate
- [ ] Monitor OpenAI costs
- [ ] Document API endpoints
- [ ] Create user guide

### Ashoka (Frontend: Deploy & Infrastructure Review)
- [ ] Build React app (npm run build)
- [ ] Upload to S3: `aws s3 sync build/ s3://bucket`
- [ ] Test frontend URL
- [ ] Review Terraform code, add tags
- [ ] Document infrastructure setup

---

## Week 6: Documentation (2-3 hours each)

### Both
- [ ] Final testing together
- [ ] Fix critical bugs
- [ ] Write README (overview, deployment, usage, costs)
- [ ] Create deployment runbook
- [ ] Review actual AWS and OpenAI costs
- [ ] Document lessons learned
- [ ] Plan Phase 2

---

## Monthly Cost Breakdown

| Service | Configuration | Cost/Month |
|---------|---------------|------------|
| EKS Control Plane | 1 cluster | $73.00 |
| EC2 Nodes | 2x t3.medium | $60.74 |
| ElastiCache Redis | cache.t3.micro | $12.41 |
| EBS | 3x 20GB gp3 | $6.00 |
| S3 | 10GB + requests | $0.33 |
| CloudWatch | Logs + metrics | $10.15 |
| Data Transfer | 10GB out | $0.90 |
| **AWS Total** | | **$163.53** |
| OpenAI API | Light usage | $17.00 |
| **GRAND TOTAL** | | **$180.53** |

---

## Cost Optimization (if needed)

| Change | Savings | Trade-off |
|--------|---------|-----------|
| Use 1 node instead of 2 | -$30/mo | Less reliability |
| Skip Redis, use in-memory | -$12/mo | Cache lost on restart |
| Use t3.small nodes | -$20/mo | Lower performance |
| **Optimized Total** | **~$118/mo** | Acceptable for MVP |

---

## Risk Mitigation

### Cost Control
- Set billing alerts at $50, $100, $150, $200
- Check AWS Cost Explorer weekly
- Set OpenAI hard limit at $50/month
- Tag all resources: `project:rag-mvp`

### Emergency Shutdown
If costs spike above $250:
1. Stop EKS nodes: `aws eks update-nodegroup-config --scaling-config minSize=0`
2. Check Cost Explorer for cause
3. Delete Redis if needed: `terraform destroy -target=redis`

---

## Success Metrics

| Metric | Target |
|--------|--------|
| API Latency (p95) | < 3 seconds |
| Cache Hit Rate | > 50% |
| Document Processing | < 2 min/PDF |
| Monthly AWS Cost | < $200 |
| Monthly OpenAI Cost | < $20 |
| Uptime | > 95% |

---

## Project Structure

```
rag-mvp/
├── terraform/
│   └── environments/dev/
│       ├── main.tf
│       ├── eks.tf
│       ├── s3.tf
│       ├── redis.tf
│       └── outputs.tf
├── k8s/
│   ├── qdrant/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── backend/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       └── secret.yaml
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py
│   │   ├── services/
│   │   └── models/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/
│       ├── services/
│       └── App.js
└── README.md
```

---

## Key Endpoints

**Backend API:**
- `GET /health` - Health check
- `POST /api/upload` - Upload PDF document
- `POST /api/ask` - Query with question

**Request/Response:**
```json
// POST /api/ask
Request: {"query": "What is...?"}
Response: {
  "answer": "...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "cached": false
}
```

---

## Deliverables by Week

| Week | Abhijeet | Ashoka |
|------|----------|--------|
| 0 | AWS setup (DevOps) | OpenAI + Qdrant setup (AI/Dev) |
| 1 | EKS + Monitoring (DevOps) | Qdrant on K8s + FastAPI (DevOps/Backend) |
| 2 | Upload + Embeddings (Backend/Data Eng) | S3 + Redis + Logs (DevOps) |
| 3 | Query + Cache (Backend/Data Eng) | Deploy FastAPI (DevOps) |
| 4 | React UI (Frontend) | Frontend Infra + Monitoring (DevOps/Frontend) |
| 5 | Testing + Validation (Data Eng) | Deploy Frontend (DevOps/Frontend) |
| 6 | Documentation (All) | Documentation (All) |

**Learning Coverage:**
- **Abhijeet:** DevOps (Weeks 0,1) → Backend/Data Eng (Weeks 2,3) → Frontend (Week 4) → Data Eng (Week 5)
- **Ashoka:** AI/Dev Setup (Week 0) → DevOps/Backend (Week 1) → DevOps (Week 2) → DevOps (Week 3) → DevOps/Frontend (Week 4,5)

---

## Next Steps After MVP

**Phase 2 Features (if continuing):**
- User authentication (Cognito)
- CI/CD pipeline (GitLab)
- Custom VPC with private subnets
- Multi-AZ deployment
- Auto-scaling
- Production security hardening
- Cost optimization review

---

**Start Week 0 when ready!**
