# Contributing to Cursey

Thank you for your interest in contributing to Cursey! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Camera access for testing
- CUDA support (recommended for Detectron2)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/cursey.git
   cd cursey
   ```

2. **Install Dependencies**
   ```bash
   python install_detectron2.py
   ```

3. **Run Tests**
   ```bash
   python test_detectron_implementation.py
   ```

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include system information (OS, Python version, etc.)
- Attach relevant logs or screenshots

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Consider implementation complexity

### Code Contributions

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python test_detectron_implementation.py
   ```

4. **Submit a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Ensure all tests pass

## Code Style

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

### File Organization
```
src/
├── detectron_detector.py      # Detectron2 integration
├── neumorphism_ui.py          # Modern UI components
├── detectron_main_app.py      # Main application
├── enhanced_detector.py       # Enhanced detection algorithms
├── high_performance_detector.py # High-performance detection
├── yolo_face_detector.py      # YOLO-style detection
├── eye_detector.py           # Eye detection module
├── gaze_analyzer.py          # Gaze analysis
├── constants.py              # Application constants
└── __init__.py               # Package initialization
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `DetectronFaceDetector`)
- Functions/Variables: `snake_case` (e.g., `process_frame`)
- Constants: `UPPER_CASE` (e.g., `MAX_PERSONS`)
- Files: `snake_case` (e.g., `detectron_detector.py`)

## Testing

### Test Structure
- Unit tests for individual components
- Integration tests for system functionality
- Performance tests for optimization validation

### Running Tests
```bash
# Run all tests
python test_detectron_implementation.py

# Run specific detector tests
python -m pytest tests/test_detector.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Documentation

### Code Documentation
- Use Google-style docstrings
- Include parameter descriptions
- Provide usage examples
- Document return values and exceptions

### Example Docstring
```python
def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
    """
    Process a single frame with Detectron2 face detection.
    
    Args:
        frame: Input frame from camera (BGR format)
        
    Returns:
        Dictionary containing:
            - persons: Dict of detected persons with IDs
            - frame: Processed frame
            - detection_count: Number of detections
            - performance_stats: Performance metrics
            
    Raises:
        ValueError: If frame is invalid
        RuntimeError: If detection fails
    """
```

## Development Guidelines

### Performance Considerations
- Optimize for real-time processing (30+ FPS)
- Use GPU acceleration when available
- Implement efficient memory management
- Profile code for bottlenecks

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Implement graceful fallbacks
- Log errors appropriately

### UI/UX Guidelines
- Follow neumorphism design principles
- Ensure responsive design
- Provide clear visual feedback
- Maintain consistent color schemes

## Debugging

### Common Issues
1. **Detectron2 Installation**: Use the provided installation script
2. **CUDA Issues**: Check GPU compatibility and drivers
3. **Camera Access**: Verify permissions and device availability
4. **Performance**: Monitor system resources and optimize accordingly

### Debug Tools
- Use logging for debugging information
- Implement performance profiling
- Add visual debugging overlays
- Create diagnostic utilities

## Pull Request Process

1. **Pre-submission Checklist**
   - [ ] Code follows style guidelines
   - [ ] Tests pass successfully
   - [ ] Documentation is updated
   - [ ] No breaking changes (or clearly documented)

2. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   ```

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow professional communication standards

### Getting Help
- Check existing issues and discussions
- Ask questions in GitHub discussions
- Join our community channels
- Contribute to documentation improvements

## License

By contributing to Cursey, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Cursey!
