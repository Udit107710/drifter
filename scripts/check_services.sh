#!/usr/bin/env bash
# Check connectivity to docker-compose services.
# Usage: ./scripts/check_services.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

check_http() {
    local name="$1" url="$2"
    if curl -sf --max-time 3 "$url" > /dev/null 2>&1; then
        echo -e "  ${GREEN}[OK]${NC}  $name ($url)"
        return 0
    else
        echo -e "  ${RED}[FAIL]${NC} $name ($url)"
        return 1
    fi
}

check_tcp() {
    local name="$1" host="$2" port="$3"
    if timeout 3 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC}  $name ($host:$port)"
        return 0
    else
        echo -e "  ${RED}[FAIL]${NC} $name ($host:$port)"
        return 1
    fi
}

echo "Checking docker-compose services..."
echo ""

failures=0

echo "Infrastructure services:"
check_http "Qdrant"       "http://localhost:6333/healthz"              || ((failures++))
check_http "OpenSearch"   "http://localhost:9200/_cluster/health"      || ((failures++))
check_tcp  "Postgres"     "localhost" 5432                             || ((failures++))
check_http "MinIO"        "http://localhost:9000/minio/health/live"    || ((failures++))

echo ""
echo "Observability:"
check_http "Jaeger UI"    "http://localhost:16686/"                    || ((failures++))
check_tcp  "OTLP HTTP"   "localhost" 4318                             || ((failures++))
check_tcp  "OTLP gRPC"   "localhost" 4317                             || ((failures++))

echo ""
echo "Model-serving (optional, not in docker-compose):"
check_http "TEI"          "http://localhost:8080/health"               || true
check_http "vLLM"         "http://localhost:8000/health"               || true

echo ""
if [ "$failures" -eq 0 ]; then
    echo -e "${GREEN}All infrastructure services are healthy.${NC}"
else
    echo -e "${YELLOW}$failures service(s) not reachable.${NC}"
    echo "Run: docker compose up -d"
fi
