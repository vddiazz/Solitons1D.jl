#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: $0 \"your commit message\""
  exit 1
fi

ssh git@github.com

git remote set-url origin git@github.com:vddiazz/Solitons1D.jl.git

git add .
git commit -m "$1"
git push
