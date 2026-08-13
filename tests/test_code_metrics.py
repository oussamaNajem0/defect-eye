from pathlib import Path
import pytest
from defect_eye.extractor.code_metrics import analyze_file, file_metrics_to_dict


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    code = '''
def simple_function(a, b):
    return a + b

def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            return x + y
        elif z > 0:
            return x + z
    else:
        for i in range(10):
            print(i)
    return 0
'''
    file = tmp_path / "sample.py"
    file.write_text(code)
    return file


@pytest.fixture
def sample_cpp_file(tmp_path: Path) -> Path:
    code = '''
#include <iostream>

int calculate(int a, int b, int c) {
    if (a > 0 && b > 0) {
        return a + b;
    }
    return c;
}

void printHello() {
    std::cout << "Hello World";
}
'''
    file = tmp_path / "sample.cpp"
    file.write_text(code)
    return file


@pytest.fixture
def sample_java_file(tmp_path: Path) -> Path:
    code = '''
public class Processor {
    public int processData(int value, boolean flag, String name) {
        if (flag) {
            switch(value) {
                case 1: return 10;
                case 2: return 20;
                default: return 0;
            }
        }
        return -1;
    }
}
'''
    file = tmp_path / "Processor.java"
    file.write_text(code)
    return file


def test_analyze_python_file(sample_python_file: Path):
    metrics = analyze_file(sample_python_file)
    assert metrics is not None
    assert metrics.function_count == 2
    assert metrics.max_cyclomatic_complexity >= 4  # Complex function has nested logic
    assert metrics.max_parameter_count == 3


def test_analyze_cpp_file(sample_cpp_file: Path):
    metrics = analyze_file(sample_cpp_file)
    assert metrics is not None
    assert metrics.function_count == 2
    assert metrics.max_parameter_count == 3


def test_analyze_java_file(sample_java_file: Path):
    metrics = analyze_file(sample_java_file)
    assert metrics is not None
    assert metrics.function_count == 1
    assert metrics.max_cyclomatic_complexity >= 4  # Switch statement increases CC


def test_file_metrics_to_dict(sample_python_file: Path):
    metrics = analyze_file(sample_python_file)
    data = file_metrics_to_dict(metrics)
    assert isinstance(data, dict)
    assert "max_cyclomatic_complexity" in data
    assert "nloc" in data


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyze_file("non_existent_file.py")