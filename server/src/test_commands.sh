# Quick manual checks against retrieval_server_visual (default ./start.sh port 8001)
curl -s http://localhost:8001/health | python3 -m json.tool

curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red car", "top_k": 5}' | head -c 500
