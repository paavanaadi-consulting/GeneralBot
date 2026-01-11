# Security Policy

## Security Considerations for Production Deployment

GeneralBot is designed with security in mind, but additional measures should be taken for production deployments.

### 1. API Security

#### Authentication & Authorization
- **Current State**: No authentication implemented (development only)
- **Production Recommendation**: Implement one of the following:
  - API Key authentication
  - OAuth 2.0 / OpenID Connect
  - JWT tokens
  - Role-Based Access Control (RBAC)

#### CORS Configuration
- **Current State**: Allows all origins (`allow_origins=["*"]`)
- **Production Recommendation**: 
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://yourdomain.com"],  # Specific domains only
      allow_credentials=True,
      allow_methods=["GET", "POST", "DELETE"],  # Only needed methods
      allow_headers=["*"],
  )
  ```

#### Rate Limiting
- **Current State**: No rate limiting
- **Production Recommendation**: Implement rate limiting to prevent abuse
  ```python
  from slowapi import Limiter
  from slowapi.util import get_remote_address
  
  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter
  
  @app.post("/chat")
  @limiter.limit("10/minute")
  async def chat(request: Request, chat_request: ChatRequest):
      ...
  ```

### 2. API Keys & Secrets

#### Environment Variables
- ✅ API keys stored in environment variables
- ✅ `.env` file in `.gitignore`
- ⚠️ Ensure `.env` file has proper permissions (600)
- ⚠️ Use secret management services in production:
  - AWS Secrets Manager
  - Azure Key Vault
  - HashiCorp Vault
  - Google Cloud Secret Manager

#### Best Practices
```bash
# Set proper permissions
chmod 600 .env

# Use different keys for development/staging/production
OPENAI_API_KEY_DEV=sk-...
OPENAI_API_KEY_PROD=sk-...
```

### 3. Input Validation

#### Current Measures
- ✅ Pydantic models for request validation
- ✅ File type validation in document processor
- ✅ Path validation for file operations

#### Additional Recommendations
- Implement file size limits for uploads
- Validate file content (not just extension)
- Sanitize user inputs to prevent injection attacks
- Implement query length limits

### 4. File Upload Security

#### Current Implementation
- ✅ Temporary file storage
- ✅ Cleanup after processing
- ⚠️ No file size limits

#### Production Recommendations
```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
    
    # Validate file type
    allowed_types = ['application/pdf', 'text/plain', ...]
    if file.content_type not in allowed_types:
        raise HTTPException(400, "Invalid file type")
```

### 5. Data Privacy

#### Conversation Data
- **Current State**: In-memory session storage
- **Concerns**: 
  - Data lost on server restart
  - No encryption at rest
- **Recommendations**:
  - Implement persistent storage with encryption
  - Regular data cleanup policies
  - User consent for data retention
  - Comply with GDPR/CCPA requirements

#### Document Storage
- **Current State**: Vector database stored on disk
- **Recommendations**:
  - Encrypt vector database at rest
  - Implement access controls
  - Regular backups
  - Secure deletion when needed

### 6. Network Security

#### HTTPS/TLS
- **Production Requirement**: Always use HTTPS
- Use certificates from trusted CAs
- Enable HTTP Strict Transport Security (HSTS)

```python
# Add security headers
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com"])

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### 7. Logging & Monitoring

#### Current Implementation
- ✅ Structured JSON logging
- ✅ No sensitive data in logs
- ✅ Error tracking

#### Production Enhancements
- Implement log aggregation (ELK, Splunk)
- Set up security monitoring and alerts
- Regular security audits
- Intrusion detection

### 8. Dependency Management

#### Best Practices
```bash
# Regular dependency updates
pip list --outdated

# Security vulnerability scanning
pip install safety
safety check

# Or use
pip-audit
```

#### Recommendations
- Pin dependency versions in production
- Regular security updates
- Automated vulnerability scanning in CI/CD

### 9. Docker Security

#### Current Dockerfile Improvements Needed
```dockerfile
# Use specific version tags
FROM python:3.10.12-slim

# Run as non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Limit container capabilities
# docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE ...

# Use multi-stage builds to reduce attack surface
FROM python:3.10-slim as builder
...
FROM python:3.10-slim
COPY --from=builder /app /app
```

### 10. OpenAI API Security

#### Best Practices
- Use separate API keys for different environments
- Implement spending limits in OpenAI dashboard
- Monitor API usage regularly
- Rotate API keys periodically
- Use Azure OpenAI for enterprise compliance needs

### 11. Vector Database Security

#### ChromaDB Considerations
- Access control if exposed over network
- Encrypt at rest
- Regular backups
- Secure connection strings

### 12. Session Management

#### Recommendations
- Implement session timeouts
- Use secure session IDs (cryptographically random)
- Clear expired sessions regularly
- Implement logout functionality

```python
import secrets

def generate_session_id():
    return secrets.token_urlsafe(32)
```

## Vulnerability Reporting

If you discover a security vulnerability, please email: security@generalbot.com

**Please do not report security vulnerabilities through public GitHub issues.**

## Security Checklist for Production

- [ ] Enable HTTPS/TLS
- [ ] Implement authentication
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Set up secret management
- [ ] Implement file size limits
- [ ] Add security headers
- [ ] Set up monitoring and alerting
- [ ] Regular security audits
- [ ] Dependency vulnerability scanning
- [ ] Run as non-root user in Docker
- [ ] Implement session management
- [ ] Enable encryption at rest
- [ ] Set up backup procedures
- [ ] Configure firewall rules
- [ ] Implement logging without sensitive data
- [ ] Regular API key rotation
- [ ] Data retention policies
- [ ] Incident response plan

## Compliance

For regulated industries, consider:
- HIPAA compliance (healthcare)
- PCI DSS (payment data)
- GDPR (EU data)
- SOC 2 certification
- ISO 27001

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [OpenAI Security Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Docker Security](https://docs.docker.com/engine/security/)

---

Last Updated: 2026-01-11
