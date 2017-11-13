ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH=$ENV_BASE_DIR/src:$ENV_BASE_DIR/src/scripts:$PATH

unset ENV_BASE_DIR
