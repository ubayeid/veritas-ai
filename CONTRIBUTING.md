# Contributing to Veritas AI

Thank you for your interest in contributing to Veritas AI! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- **Clear title** describing the bug
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, etc.)
- **Error messages** or logs (if applicable)

### Suggesting Features

Feature suggestions are welcome! Please open an issue with:
- **Clear description** of the feature
- **Use case** explaining why it's useful
- **Proposed implementation** (if you have ideas)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test your changes**: Ensure all functionality works
5. **Commit your changes**: Use clear, descriptive commit messages
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Open a Pull Request**

## 📝 Code Style Guidelines

### Python Code

- Follow **PEP 8** style guide
- Use **type hints** for function parameters and return types
- Add **docstrings** to all functions and classes (Google style)
- Maximum line length: **100 characters**
- Use **4 spaces** for indentation (no tabs)

### Example

```python
def search_documents(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search documents using vector similarity.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of document dictionaries with 'text' and 'score' keys
        
    Raises:
        ValueError: If query is empty
    """
    if not query:
        raise ValueError("Query cannot be empty")
    
    # Implementation here
    return results
```

### Commit Messages

Use clear, descriptive commit messages:

- ✅ `Add hybrid search mode to CLI`
- ✅ `Fix Neo4j connection timeout issue`
- ✅ `Update README with installation instructions`
- ❌ `fix stuff`
- ❌ `update`
- ❌ `WIP`

### Documentation

- Update README.md if adding new features
- Add docstrings to all new functions/classes
- Update relevant documentation files
- Include examples in docstrings

## 🧪 Testing

Before submitting a PR:

1. **Test your changes**:
   ```bash
   python query.py interactive --mode hybrid
   ```

2. **Check for errors**:
   - Run the code and verify it works
   - Check for linting errors
   - Ensure imports work correctly

3. **Test edge cases**:
   - Empty inputs
   - Invalid inputs
   - Error handling

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All functions have docstrings
- [ ] Code is tested and works
- [ ] README/documentation is updated (if needed)
- [ ] Commit messages are clear
- [ ] No sensitive data (API keys, passwords) in code
- [ ] Changes are focused (one feature/fix per PR)

## 🏗️ Project Structure

Understanding the project structure helps with contributions:

- `backend/agents/` - Multi-agent architecture
- `backend/retrieval/` - Query engines
- `backend/indexing/` - Database builders
- `backend/processing/` - Data processing
- `backend/evaluation/` - Evaluation framework
- `frontend/` - Web interface
- `query.py` - Main CLI entry point

## 🐛 Debugging Tips

### Common Issues

1. **Import Errors**: Ensure you're running from project root
2. **Neo4j Connection**: Check Neo4j is running and credentials are correct
3. **API Errors**: Verify API keys in `.env` file
4. **FAISS Index Errors**: Ensure indexes are built before querying

### Getting Help

- Check existing [GitHub Issues](https://github.com/ubayeid/veritas-ai/issues)
- Review documentation in `docs/` directory
- Open a new issue if needed

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make Veritas AI better for everyone. Thank you for taking the time to contribute!
