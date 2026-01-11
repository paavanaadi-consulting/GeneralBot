# Data Directory

Storage for training data, processed documents, and embeddings.

## Structure

- `raw/` - Original, unprocessed documents and data files
- `processed/` - Cleaned and preprocessed data ready for embedding
- `embeddings/` - Generated vector embeddings for RAG system

## Important Notes

- Add large data files to `.gitignore`
- Consider using Git LFS for versioning large datasets
- Implement proper data governance and privacy controls
