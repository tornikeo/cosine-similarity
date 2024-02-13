try:
    # Python <= 3.8
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
from pytest_notebook.nb_regression import NBRegressionFixture
import os

def test_notebook():
    os.environ['TQDM_DISABLE'] = '1'
    fixture = NBRegressionFixture(exec_timeout=200)
    fixture.force_regen = True
    fixture.check(str("notebooks/tutorial_pesticide.ipynb"))
