#!/bin/bash

# Exit on error
set -e

echo "Running database migrations..."
# Add any database migration commands here if needed

echo "Installing dependencies..."
pip install -r requirements.txt

echo "BugXpert deployment complete!" 