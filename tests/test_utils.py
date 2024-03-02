
import pytest
import os
@pytest.mark.skipif(
    os.getenv('GITHUB_WORKFLOW') == '1',
    reason="Github workflow doesn't have a GPU"
)
def test_get_gpu_name():
    from cudams.utils import get_device_name_as_str
    gpu_names = get_device_name_as_str()
    print("GPU Device Names:")
    for name in gpu_names:
        print(name)
        assert isinstance(name, str)