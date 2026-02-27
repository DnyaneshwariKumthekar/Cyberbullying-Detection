"""Helper to load `scripts/run_pipeline.py` and call check_dependencies() safely."""
import importlib.util
spec = importlib.util.spec_from_file_location('rp', 'scripts/run_pipeline.py')
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
print('\n--- check_dependencies result ---')
print(rp.check_dependencies())
