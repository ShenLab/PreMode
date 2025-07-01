#!/bin/bash

# Script to remove files listed in .gitignore from a specific commit
# Usage: ./remove_files_from_commit.sh <commit_id>

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

echo "Creating backup and removing .gitignore files from commit: $COMMIT_ID"

# Create a backup branch
BACKUP_BRANCH="backup-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
echo "Created backup branch: $BACKUP_BRANCH"

# Set environment variable to suppress warning
export FILTER_BRANCH_SQUELCH_WARNING=1

# Get list of files to remove from current .gitignore
TEMP_FILE=$(mktemp)
grep -v "^#" .gitignore | grep -v "^$" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' > "$TEMP_FILE"

echo "Files patterns to remove:"
cat "$TEMP_FILE"
echo ""

# Use filter-branch to remove files
git filter-branch --force --index-filter "
while IFS= read -r pattern; do
    if [[ -n \"\$pattern\" ]]; then
        git rm --cached --ignore-unmatch -r \"\$pattern\" 2>/dev/null || true
    fi
done < \"$TEMP_FILE\"
" --prune-empty HEAD

# Clean up
rm -f "$TEMP_FILE"

echo ""
echo "Filter completed!"
echo "Backup branch created: $BACKUP_BRANCH"
echo ""
echo "To restore if something went wrong:"
echo "git reset --hard $BACKUP_BRANCH"
echo ""
echo "To delete backup after confirming everything is OK:"
echo "git branch -D $BACKUP_BRANCH"