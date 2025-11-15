# Jupyter Notebook Collaboration Setup

This project uses **nbdime** to handle Jupyter notebook diffs and merges in git. This prevents merge conflicts and makes collaboration easier.

## Quick Setup (5 minutes)

### 1. Install nbdime

```bash
pip install nbdime
```

### 2. Configure git (One-time setup)

After cloning this repo, run:

```bash
cd DL_comp3
nbdime config-git --enable
```

That's it! Git will now use nbdime automatically for `.ipynb` files.

## Verify Setup

Check that it worked:

```bash
git config --local --get diff.jupyternotebook.command
# Should output: git-nbdiffdriver diff
```

## Usage

### Normal Workflow
```bash
# Work on notebook as usual
jupyter notebook template.ipynb

# Git operations work normally
git diff template.ipynb          # Clean, readable diff!
git add template.ipynb
git commit -m "Update model"
git push
```

### Handling Merge Conflicts

If you encounter a merge conflict:

```bash
git pull
# If conflict occurs:
git mergetool                    # Uses nbdime automatically

# Or use visual merge tool:
nbdime mergetool
```

### Visual Diff Tool (Optional)

Compare notebooks visually in browser:

```bash
nbdiff-web template.ipynb                    # Show changes
nbdiff-web old.ipynb new.ipynb              # Compare two versions
```

## Optional: Auto-strip Outputs

To reduce conflicts further, you can auto-strip cell outputs before commits:

1. Uncomment line 5 in `.gitattributes`:
   ```
   *.ipynb filter=nbstripout
   ```

2. Install nbstripout:
   ```bash
   pip install nbstripout
   nbstripout --install
   ```

**Note:** This removes all cell outputs from git, keeping only code and markdown.

## Troubleshooting

### "No notebook detected" error
If you see errors about filters, make sure you ran:
```bash
nbdime config-git --enable
```

### Conflicts still happening
Make sure everyone on the team has nbdime installed and configured!

## What This Does

- **Before:** Merge conflicts in JSON = 😭
- **After:** Smart notebook-aware merging = 😊

nbdime understands notebook structure and merges:
- Code cells
- Markdown cells
- Outputs
- Metadata

...intelligently, instead of treating notebooks as plain JSON.

## Questions?

See: https://nbdime.readthedocs.io/
