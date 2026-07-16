from common.logger import LoggerManager
from common.seed import set_seed

LoggerManager()

set_seed(42)

print("Seed configured successfully.")