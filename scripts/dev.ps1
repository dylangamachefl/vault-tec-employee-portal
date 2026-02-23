param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "test", "docker-up", "docker-down", "lint")]
    [string]$Command
)

switch ($Command) {
    "setup" {
        uv sync
        uv run pre-commit install
    }
    "docker-up" {
        docker compose up -d
    }
    "docker-down" {
        docker compose down
    }
    "test" {
        uv run pytest tests/ -v
    }
    "lint" {
        uv run ruff check src/ --fix
    }
}
