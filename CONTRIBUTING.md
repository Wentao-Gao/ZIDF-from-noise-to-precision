# Contributing to ZIDF

Thank you for your interest in contributing to ZIDF! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the [Issues](https://github.com/Wentao-Gao/ZIDF-from-noise-to-precision/issues) section
2. If not, create a new issue with a clear title and description
3. Include steps to reproduce the bug (if applicable)
4. Add relevant labels

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Comment complex logic

### Testing

- Add unit tests for new features
- Ensure all existing tests pass
- Test with different configurations

### Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update example notebooks if functionality changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ZIDF-from-noise-to-precision.git
cd ZIDF-from-noise-to-precision

# Create virtual environment
conda create -n zidf python=3.9
conda activate zidf

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Review Process

1. All submissions require review
2. We may suggest changes or improvements
3. Once approved, your PR will be merged

## Community Guidelines

- Be respectful and constructive
- Help others in discussions
- Follow the code of conduct

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing!
