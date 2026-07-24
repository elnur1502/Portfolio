#!/bin/bash

echo "Waiting 15 seconds for Langfuse and Postgres to fully initialize..."
sleep 15

cd /app
exec python3 main.py