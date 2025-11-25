#!/bin/bash
# Helper script to update .env with Docker Neo4j settings

BOLT_PORT=$(docker port neo4j 2>/dev/null | grep 7687 | cut -d: -f2 || echo "7689")

if [ -f .env ]; then
    # Update existing .env
    if grep -q "^NEO4J_URI=" .env; then
        sed -i "s|^NEO4J_URI=.*|NEO4J_URI=bolt://localhost:$BOLT_PORT|" .env
    else
        echo "NEO4J_URI=bolt://localhost:$BOLT_PORT" >> .env
    fi
    
    if grep -q "^NEO4J_USER=" .env; then
        sed -i "s|^NEO4J_USER=.*|NEO4J_USER=neo4j|" .env
    else
        echo "NEO4J_USER=neo4j" >> .env
    fi
    
    if grep -q "^NEO4J_PASSWORD=" .env; then
        sed -i "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=password|" .env
    else
        echo "NEO4J_PASSWORD=password" >> .env
    fi
    
    echo "✓ .env file updated!"
else
    echo "Creating .env file..."
    echo "NEO4J_URI=bolt://localhost:$BOLT_PORT" > .env
    echo "NEO4J_USER=neo4j" >> .env
    echo "NEO4J_PASSWORD=password" >> .env
    echo "✓ .env file created!"
fi

echo ""
echo "Current Neo4j Docker settings:"
grep "^NEO4J_" .env

