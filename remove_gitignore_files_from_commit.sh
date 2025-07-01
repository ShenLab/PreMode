#!/bin/bash

# Script to remove files listed in .gitignore from a specific commit
# Usage: ./remove_gitignore_files_from_commit.sh <commit_id>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <commit_id>"
    echo "Example: $0 abc123def"
    exit 1
fi

COMMIT_ID="$1"

# Verify commit exists
if ! git rev-parse --verify "$COMMIT_ID" >/dev/null 2>&1; then
    echo "Error: Commit $COMMIT_ID does not exist"
    exit 1
fi

echo "Removing .gitignore files from commit: $COMMIT_ID"

# Create a backup branch
BACKUP_BRANCH="backup-before-filter-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
echo "Created backup branch: $BACKUP_BRANCH"

# Use filter-branch to target specific commit and remove .gitignore files
git filter-branch --force --index-filter '
# Only process if this is the target commit
if [ "$GIT_COMMIT" = "'"$COMMIT_ID"'" ]; then
    echo "Processing commit: $GIT_COMMIT"
    
    # Read .gitignore patterns and remove them
    if [ -f .gitignore ]; then
        echo "Reading .gitignore patterns..."
        while IFS= read -r pattern; do
            # Skip empty lines and comments
            if [[ -n "$pattern" && ! "$pattern" =~ ^[[:space:]]*# ]]; then
                # Remove leading/trailing whitespace
                pattern=$(echo "$pattern" | sed "s/^[[:space:]]*//;s/[[:space:]]*$//")
                if [[ -n "$pattern" ]]; then
                    echo "Removing pattern: $pattern"
                    git rm --cached --ignore-unmatch -r "$pattern" 2>/dev/null || true
                fi
            fi
        done < .gitignore
    else
        echo "No .gitignore file found in this commit"
    fi
else
    echo "Skipping commit: $GIT_COMMIT (not target)"
fi
' --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Filter completed!"
echo "Backup branch created: $BACKUP_BRANCH"
echo ""
echo "To restore if something went wrong:"
echo "git checkout $BACKUP_BRANCH"
echo ""
echo "To delete backup after confirming everything is OK:"
echo "git branch -D $BACKUP_BRANCH"